import torch
import torch.nn as nn

from transformers import AutoImageProcessor, VideoMAEModel, VideoMAEConfig
from transformers import BlipTextModel
from peft import get_peft_model


class SurgViVQA(nn.Module):
    """
    SurgViVQA = VideoMAE (video encoder) + BLIP text encoder (cross-attn to video) + LoRA-adapted LLM (e.g. GPT-2)

    IMPORTANT:
      - VideoMAEConfig.num_frames MUST match dataset num_frames.
      - BLIP word embedding MUST be resized to match the tokenizer vocab size used in training (often GPT-2 = 50257),
        otherwise loading checkpoint will size-mismatch.
    """
    def __init__(
        self,
        device=torch.device("cpu"),
        tokenizer=None,
        decoder_model=None,
        peft_config=None,
        num_frames: int = 8,
    ):
        super().__init__()
        self.device = device

        if tokenizer is None:
            raise ValueError("tokenizer must be provided (e.g., GPT-2 tokenizer).")
        self.tokenizer = tokenizer

        # 1) Visual encoder (VideoMAE)
        model_name = "MCG-NJU/videomae-base"
        print("Visual Encoder version:", model_name)

        config = VideoMAEConfig.from_pretrained(
            model_name,
            cache_dir="/scratch/sc232jl/.cache/huggingface",
        )

        # Normalize config fields to valid ints (VideoMAE expects scalar ints)
        def _as_int(x, name: str):
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    raise ValueError(f"VideoMAE config {name} is empty.")
                return int(x[0])
            return int(x)

        config.image_size = _as_int(config.image_size, "image_size")
        config.patch_size = _as_int(config.patch_size, "patch_size")
        config.tubelet_size = _as_int(config.tubelet_size, "tubelet_size")

        # Ensure num_frames is compatible with tubelet_size to avoid zero patches
        nf = int(num_frames)
        if nf < int(config.tubelet_size):
            print(
                f"[WARN] num_frames ({nf}) < tubelet_size ({config.tubelet_size}); "
                "bumping num_frames to tubelet_size for valid patch grid."
            )
            nf = int(config.tubelet_size)
        config.num_frames = nf

        # Log effective config values before model construction (for debugging)
        print(
            "[VideoMAEConfig] image_size=", config.image_size,
            "patch_size=", config.patch_size,
            "tubelet_size=", config.tubelet_size,
            "num_frames=", config.num_frames,
            "hidden_size=", config.hidden_size,
        )

        self.num_frames = int(config.num_frames)

        self.visual_encoder = VideoMAEModel.from_pretrained(
            model_name,
            config=config,
            cache_dir="/scratch/sc232jl/.cache/huggingface",
            ignore_mismatched_sizes=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

        # Freeze video encoder params
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        # 2) Text encoder (BLIP text model with cross-attn)
        self.text_encoder = BlipTextModel.from_pretrained(
            "Salesforce/blip-vqa-base",
            cache_dir="/scratch/sc232jl/.cache/huggingface",
        )

        # ---- CRITICAL: resize BLIP word embeddings to match tokenizer vocab size ----
        self._resize_blip_word_embeddings(new_vocab_size=len(self.tokenizer))

        # 3) LLM decoder (LoRA optional)
        if decoder_model is None:
            raise ValueError("decoder_model is required, e.g. AutoModelForCausalLM.from_pretrained('gpt2').")
        self.llm = decoder_model
        if peft_config is not None:
            self.llm = get_peft_model(self.llm, peft_config)

    def _resize_blip_word_embeddings(self, new_vocab_size: int):
        """
        Expand BLIP text encoder's word_embeddings from original vocab (e.g., 30524) to new_vocab_size (e.g., 50257).
        This must happen BEFORE loading the fine-tuned checkpoint, otherwise state_dict size mismatch occurs.
        """
        if not hasattr(self.text_encoder, "embeddings") or not hasattr(self.text_encoder.embeddings, "word_embeddings"):
            raise RuntimeError("Unexpected BLIP text encoder structure: cannot find embeddings.word_embeddings.")

        old_emb: nn.Embedding = self.text_encoder.embeddings.word_embeddings
        old_vocab_size, hidden = old_emb.weight.shape

        if int(new_vocab_size) == int(old_vocab_size):
            return

        if new_vocab_size < old_vocab_size:
            print(f"[WARN] new_vocab_size ({new_vocab_size}) < old_vocab_size ({old_vocab_size}), truncating embeddings.")
            new_emb = nn.Embedding(new_vocab_size, hidden)
            with torch.no_grad():
                new_emb.weight.copy_(old_emb.weight[:new_vocab_size])
            self.text_encoder.embeddings.word_embeddings = new_emb
            if hasattr(self.text_encoder, "config") and hasattr(self.text_encoder.config, "vocab_size"):
                self.text_encoder.config.vocab_size = int(new_vocab_size)
            return

        print(f"[INFO] Resizing BLIP word_embeddings: {old_vocab_size} -> {new_vocab_size}")
        new_emb = nn.Embedding(new_vocab_size, hidden)
        with torch.no_grad():
            new_emb.weight[:old_vocab_size].copy_(old_emb.weight)
            old_mean = old_emb.weight.mean().item()
            old_std = old_emb.weight.std().item()
            new_emb.weight[old_vocab_size:].normal_(mean=old_mean, std=old_std)

        self.text_encoder.embeddings.word_embeddings = new_emb
        if hasattr(self.text_encoder, "config") and hasattr(self.text_encoder.config, "vocab_size"):
            self.text_encoder.config.vocab_size = int(new_vocab_size)

    def encode_video(self, video: torch.Tensor):
        """
        video: [B, T, 3, 224, 224]
        returns:
          video_embeds: [B, V, D]
          video_atts:   [B, V]
        """
        # If a single frame is provided, repeat to a pseudo-video of length self.num_frames
        # This avoids VideoMAE crashes when num_frames < tubelet_size and ensures valid patch grids.
        if video.dim() == 4:
            video = video.unsqueeze(1)  # [B, 1, 3, H, W]
        if video.size(1) == 1 and self.num_frames > 1:
            video = video.repeat(1, self.num_frames, 1, 1, 1)

        out = self.visual_encoder(pixel_values=video)
        video_embeds = out.last_hidden_state
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long, device=video_embeds.device)
        return video_embeds, video_atts

    def forward(self, video, qa_inputs_ids, qa_att_mask, video_embeds=None, video_atts=None):
        """
        If video_embeds/video_atts are provided (as in closed decoding scoring),
        we skip encode_video for speed.
        """
        if video_embeds is None or video_atts is None:
            video_embeds, video_atts = self.encode_video(video)

        text_out = self.text_encoder(
            input_ids=qa_inputs_ids,
            attention_mask=qa_att_mask,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        text_embeds = text_out.last_hidden_state

        llm_out = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=qa_att_mask,
        )
        return llm_out.logits
