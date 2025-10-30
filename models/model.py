import torch
import torch.nn as nn

from transformers import AutoImageProcessor, VideoMAEModel, VideoMAEConfig  # VideoMAE video encoder
from transformers import BlipTextModel

from peft import get_peft_model


class SurgViVQA(nn.Module):
    """
    SurgViVQA: Masked Video–Text Encoder + LoRA-adapted LLM for surgical video VQA.

    Architecture:
    1. Video Encoder (VideoMAE) → Extracts spatiotemporal embeddings from video clips
    2. Text Encoder (BLIP) → Encodes questions, cross-attends to video embeddings
    3. LoRA-adapted LLM → Autoregressively generates free-form answers
    """
    def __init__(self, device=torch.device('cpu'), tokenizer=None, decoder_model=None, peft_config=None):
        super(SurgViVQA, self).__init__()
        
        self.device = device

        model_name = "MCG-NJU/videomae-base"  # change to desired HF VideoMAE checkpoint
        print("Visual Encoder version: ", model_name)

        config = VideoMAEConfig.from_pretrained(
            model_name,
            cache_dir="/SAN/medic/Kvasir/hf_models/transformers"
        )
        config.num_frames = 8

        self.visual_encoder = VideoMAEModel.from_pretrained(
            model_name,
            config=config,
            cache_dir="/SAN/medic/Kvasir/hf_models/transformers",
            ignore_mismatched_sizes=True,
        )

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

        # Freeze video encoder params
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        print("Frozen Visual Encoder (VideoMAE)")

        # Unfreeze patch embedding (Conv3d)
        for param in self.visual_encoder.embeddings.patch_embeddings.projection.parameters():
            param.requires_grad = True

        # Unfreeze final layernorm
        for param in self.visual_encoder.layernorm.parameters():
            param.requires_grad = True

        # tokenizer
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token  # EOS used as padding

        # text encoder (BLIP)
        self.text_encoder = BlipTextModel.from_pretrained("Salesforce/blip-vqa-base", ignore_mismatched_sizes=True)

        # --- Expand BLIP embedding layer to match tokenizer ---
        original_weights = self.text_encoder.embeddings.word_embeddings.weight.data
        new_vocab_size = len(self.tokenizer)
        blip_internal_emb_dim = self.text_encoder.embeddings.word_embeddings.embedding_dim
        new_embeddings = nn.Embedding(new_vocab_size, blip_internal_emb_dim)
        original_vocab_size = original_weights.shape[0]
        new_embeddings.weight.data[:original_vocab_size] = original_weights
        self.text_encoder.embeddings.word_embeddings = new_embeddings

        # decoder
        self.decoder_model = decoder_model
        self.llm = get_peft_model(decoder_model, peft_config)
        self.llm.print_trainable_parameters()  # Verify trainable LoRA params


    def forward(self, video, qa_inputs_ids, qa_att_mask):

        # VIDEO FRAMES HANDLER CODE (sequences of images)
        # video: [batch, frames, 3, 224, 224]
        video = video.to(self.device)
        video_embeds = self.visual_encoder(pixel_values=video).last_hidden_state  # (batch, num_patches+1, hidden_dim)

        video_atts = torch.ones(video_embeds.size()[:-1],
                                dtype=torch.long,
                                device=video.device)

        # multimodal encoder (BLIP text with cross-attention over video tokens)
        text_output = self.text_encoder(
            input_ids=qa_inputs_ids,
            attention_mask=qa_att_mask,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True
        )
        text_embeds = text_output.last_hidden_state
        
        # text decoder (LoRA-adapted LLM)
        llm_output = self.llm(
            inputs_embeds=text_embeds,
            encoder_attention_mask=qa_att_mask
        )
        
        return llm_output.logits
