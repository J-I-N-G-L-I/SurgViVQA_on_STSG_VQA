"""
SurgViVQA evaluation on SSGVQA static QA (closed-set label scoring).

Key goals:
- Ensure images tensor shape is [B, T, 3, H, W].
- Support prompt modes: simple vs choices.
- Closed-set scoring over SSGVQA labels (no free-form generation).
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

# Ensure repo root is on PYTHONPATH when running as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.model import SurgViVQA


# -----------------------------
# Label vocabulary (baseline)
# -----------------------------

SSGVQA_LABELS: List[str] = [
    "0",
    "1",
    "10",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "False",
    "True",
    "abdominal_wall_cavity",
    "adhesion",
    "anatomy",
    "aspirate",
    "bipolar",
    "blood_vessel",
    "blue",
    "brown",
    "clip",
    "clipper",
    "coagulate",
    "cut",
    "cystic_artery",
    "cystic_duct",
    "cystic_pedicle",
    "cystic_plate",
    "dissect",
    "fluid",
    "gallbladder",
    "grasp",
    "grasper",
    "gut",
    "hook",
    "instrument",
    "irrigate",
    "irrigator",
    "liver",
    "omentum",
    "pack",
    "peritoneum",
    "red",
    "retract",
    "scissors",
    "silver",
    "specimen_bag",
    "specimenbag",
    "white",
    "yellow",
]


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace("_", " ")
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = " ".join(s.split())
    return s


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    exact_map = {lbl: i for i, lbl in enumerate(labels)}
    norm_map = {normalize_text(lbl): i for i, lbl in enumerate(labels)}
    return exact_map, norm_map


def map_pred_to_label(
    pred_text: str,
    labels: List[str],
    exact_map: Dict[str, int],
    norm_map: Dict[str, int],
) -> Tuple[str, int]:
    if pred_text is None:
        return "", -1
    raw = str(pred_text).strip()
    if not raw:
        return "", -1

    if raw in exact_map:
        return raw, exact_map[raw]

    raw_norm = normalize_text(raw)
    if raw_norm in norm_map:
        return labels[norm_map[raw_norm]], norm_map[raw_norm]

    if raw_norm in ("yes", "true", "y", "t") and "True" in exact_map:
        return "True", exact_map["True"]
    if raw_norm in ("no", "false", "n", "f") and "False" in exact_map:
        return "False", exact_map["False"]

    for token in raw_norm.split():
        if token.isdigit() and token in exact_map:
            return token, exact_map[token]

    for lbl in labels:
        lbl_norm = normalize_text(lbl)
        if lbl_norm and f" {lbl_norm} " in f" {raw_norm} ":
            return lbl, exact_map[lbl]

    return "", -1


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"acc": 0.0, "mAP": 0.0, "mAR": 0.0, "mAF1": 0.0, "wF1": 0.0}

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)
    acc = float(np.mean(y_true_arr == y_pred_arr))

    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(num_classes):
        tp = float(np.sum((y_true_arr == c) & (y_pred_arr == c)))
        fp = float(np.sum((y_true_arr != c) & (y_pred_arr == c)))
        fn = float(np.sum((y_true_arr == c) & (y_pred_arr != c)))
        support = float(np.sum(y_true_arr == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    mAP = float(np.mean(precisions))
    mAR = float(np.mean(recalls))
    mAF1 = float(np.mean(f1s))
    total_support = float(np.sum(supports))
    wF1 = float(np.sum(np.array(f1s) * np.array(supports)) / total_support) if total_support > 0 else 0.0

    return {"acc": acc, "mAP": mAP, "mAR": mAR, "mAF1": mAF1, "wF1": wF1}


def build_prompt_prefix(question: str, mode: str) -> str:
    q = (question or "").strip()
    if mode == "choices":
        candidates = ", ".join(SSGVQA_LABELS)
        return (
            f"Question: {q}\n"
            f"Candidates: {candidates}\n"
            "Answer:"
        )
    return (
        f"Question: {q}\n"
        "Answer:"
    )


def to_chw_frame(pixel_values: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(pixel_values):
        pixel_values = torch.tensor(pixel_values)
    if pixel_values.dim() == 5:
        # [1, 1, 3, H, W]
        return pixel_values.squeeze(0).squeeze(0)
    if pixel_values.dim() == 4:
        # [1, 3, H, W]
        return pixel_values.squeeze(0)
    if pixel_values.dim() == 3:
        # [3, H, W]
        return pixel_values
    raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")


def _tokenize_prefixes_no_pad(
    tokenizer,
    prompt_prefixes: List[str],
    max_input_tokens: int,
) -> Tuple[List[List[int]], List[int]]:
    """
    Tokenize prefixes WITHOUT padding.
    Returns:
        prefix_ids_list: List[List[int]] - token ids for each prefix
        prefix_lens: List[int] - actual length of each prefix
    """
    prefix_ids_list = []
    prefix_lens = []
    for prefix in prompt_prefixes:
        enc = tokenizer(
            prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_tokens,
        )
        ids = enc["input_ids"]
        prefix_ids_list.append(ids)
        prefix_lens.append(len(ids))
    return prefix_ids_list, prefix_lens


def _normalize_label_token_ids(tokenizer, labels: List[str]) -> List[List[int]]:
    label_ids = []
    for lbl in labels:
        text = " " + str(lbl)
        enc = tokenizer(text, add_special_tokens=False)
        label_ids.append(enc["input_ids"])
    return label_ids


def score_labels_closedset(
    model,
    tokenizer,
    videos: torch.Tensor,
    prompt_prefixes: List[str],
    labels: List[str],
    max_input_tokens: int,
    device,
    label_chunk_size: int = 10,
    score_norm: str = "mean",
):
    """
    Closed-set scoring: for each sample and each label, compute teacher-forcing log-prob.
    
    FIX: Constructs seq_ids = prefix_ids + label_ids for each sample+label pair,
    ensuring label tokens are immediately after the prefix (no pad gap).
    
    Args:
        score_norm: "sum" for raw logprob sum, "mean" for length-normalized (default).
    
    Returns:
        scores: [B, num_labels]
        tokenized_lens: list of prompt token lengths (actual, not padded)
        trunc_flags: list of bool indicating truncation
        label_tokens_scored: [B, num_labels] actual label tokens scored per sample/label
        label_token_lens: [num_labels] token length per label (for diagnostics)
    """
    # Tokenize prefixes WITHOUT padding to get actual prefix lengths
    prefix_ids_list, prefix_lens = _tokenize_prefixes_no_pad(tokenizer, prompt_prefixes, max_input_tokens)
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    video_embeds, video_atts = model.encode_video(videos.to(device))

    label_token_ids = _normalize_label_token_ids(tokenizer, labels)
    label_token_lens = [len(ids) for ids in label_token_ids]

    bsz = len(prompt_prefixes)
    num_labels = len(labels)
    scores = torch.full((bsz, num_labels), -1e9, device=device, dtype=torch.float32)
    label_tokens_scored = torch.zeros((bsz, num_labels), device=device, dtype=torch.long)
    trunc_flags = [False] * bsz

    for start in range(0, num_labels, label_chunk_size):
        end = min(start + label_chunk_size, num_labels)
        chunk_size = end - start
        chunk_label_ids = label_token_ids[start:end]
        chunk_label_lens = label_token_lens[start:end]

        # Build all sequences: for each sample i and each label j in chunk,
        # seq[i,j] = prefix_ids[i] + label_ids[j]
        all_seqs = []
        all_prefix_lens_flat = []
        all_label_lens_flat = []
        trunc_in_chunk = []
        
        for i in range(bsz):
            p_ids = prefix_ids_list[i]
            p_len = prefix_lens[i]
            for j, l_ids in enumerate(chunk_label_ids):
                l_len = chunk_label_lens[j]
                # Concatenate prefix + label (label immediately follows prefix)
                seq_ids = p_ids + l_ids
                total_len = len(seq_ids)
                
                # Check truncation
                truncated = total_len > max_input_tokens
                if truncated:
                    seq_ids = seq_ids[:max_input_tokens]
                    trunc_flags[i] = True
                
                trunc_in_chunk.append(truncated)
                all_seqs.append(torch.tensor(seq_ids, dtype=torch.long, device=device))
                all_prefix_lens_flat.append(p_len)
                all_label_lens_flat.append(l_len)
        
        # Pad sequences to same length using pad_sequence
        flat_ids = torch.nn.utils.rnn.pad_sequence(
            all_seqs, batch_first=True, padding_value=pad_id
        )  # [B*chunk_size, max_seq_len]
        
        # Build attention mask (1 for real tokens, 0 for padding)
        flat_att = (flat_ids != pad_id).long()
        
        # Expand video embeddings for flattened batch
        # video_embeds: [B, V, D] -> [B*chunk_size, V, D]
        video_embeds_exp = video_embeds.unsqueeze(1).expand(bsz, chunk_size, -1, -1).contiguous()
        video_embeds_exp = video_embeds_exp.view(bsz * chunk_size, video_embeds.size(1), video_embeds.size(2))
        video_atts_exp = video_atts.unsqueeze(1).expand(bsz, chunk_size, -1).contiguous()
        video_atts_exp = video_atts_exp.view(bsz * chunk_size, video_atts.size(1))

        with torch.no_grad():
            logits = model(
                video=None,
                qa_inputs_ids=flat_ids,
                qa_att_mask=flat_att,
                video_embeds=video_embeds_exp,
                video_atts=video_atts_exp,
            )

        log_probs = torch.log_softmax(logits, dim=-1)

        # Compute label token log-prob (teacher-forcing)
        # For position p in seq, logits[p] predicts token at p+1
        # So to score label tokens at positions [prefix_len, prefix_len + label_len),
        # we use logits at positions [prefix_len - 1, prefix_len + label_len - 1)
        seq_len = flat_ids.size(1)
        shifted_log_probs = log_probs[:, :-1, :]  # [B*chunk, seq_len-1, vocab]
        shifted_ids = flat_ids[:, 1:]  # [B*chunk, seq_len-1]

        chunk_label_scores = torch.zeros(bsz * chunk_size, device=device, dtype=torch.float32)
        chunk_tokens_scored_flat = torch.zeros(bsz * chunk_size, device=device, dtype=torch.long)

        for idx in range(bsz * chunk_size):
            p_len = all_prefix_lens_flat[idx]
            l_len = all_label_lens_flat[idx]
            
            # Label tokens are at positions [p_len, p_len + l_len) in the sequence
            # To predict token at position p_len, we use logits at position p_len - 1
            start_pos = p_len - 1  # Start of scoring window in shifted tensors
            end_pos = min(p_len - 1 + l_len, seq_len - 1)  # End of scoring window
            
            if end_pos <= start_pos or start_pos < 0:
                continue
            
            # Get the label token ids we want to score
            target_ids = shifted_ids[idx, start_pos:end_pos]
            # Create mask for valid (non-truncated) label positions
            valid_len = end_pos - start_pos
            num_valid = min(valid_len, l_len)
            
            if num_valid == 0:
                continue
            
            chunk_tokens_scored_flat[idx] = num_valid
            
            # Gather log probs for target tokens
            lp = shifted_log_probs[idx, start_pos:start_pos + num_valid, :]
            lp = lp.gather(1, target_ids[:num_valid].unsqueeze(1)).squeeze(1)
            
            sum_lp = lp.sum()
            if score_norm == "mean" and num_valid > 0:
                chunk_label_scores[idx] = sum_lp / num_valid
            else:
                chunk_label_scores[idx] = sum_lp

        chunk_label_scores = chunk_label_scores.view(bsz, chunk_size)
        chunk_tokens_scored_flat = chunk_tokens_scored_flat.view(bsz, chunk_size)
        scores[:, start:end] = chunk_label_scores
        label_tokens_scored[:, start:end] = chunk_tokens_scored_flat

    return scores, prefix_lens, trunc_flags, label_tokens_scored, label_token_lens


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class SSGVQASample:
    video_id: str
    frame_id: str
    image_path: str
    question: str
    answer: str


class SSGVQAStaticDataset(Dataset):
    def __init__(
        self,
        ssgvqa_root: str,
        image_root: str,
        video_ids: List[str],
        processor=None,
        num_frames: int = 16,
        max_samples: Optional[int] = None,
        strict_missing_image: bool = False,
    ):
        self.ssgvqa_root = ssgvqa_root
        self.image_root = image_root
        self.video_ids = video_ids
        self.processor = processor
        self.strict_missing_image = strict_missing_image
        self.num_frames = int(num_frames)
        self._log_processor_once = True

        self.samples: List[SSGVQASample] = []
        for vid in self.video_ids:
            vid_dir = os.path.join(self.ssgvqa_root, vid)
            if not os.path.isdir(vid_dir):
                raise FileNotFoundError(f"Missing SSGVQA folder: {vid_dir}")

            txt_files = [f for f in os.listdir(vid_dir) if f.endswith(".txt")]
            txt_files = sorted(txt_files, key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

            for fname in txt_files:
                frame_stem = os.path.splitext(fname)[0]
                if frame_stem.isdigit():
                    frame_id = f"{int(frame_stem):06d}"
                else:
                    frame_id = frame_stem.zfill(6)

                img_path = os.path.join(self.image_root, vid, f"{frame_id}.png")
                if not os.path.isfile(img_path):
                    if self.strict_missing_image:
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    else:
                        continue

                qa_path = os.path.join(vid_dir, fname)
                with open(qa_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "|" not in line:
                            continue
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) < 2:
                            continue
                        question = parts[0]
                        answer = parts[1]
                        if not question or not answer:
                            continue
                        self.samples.append(
                            SSGVQASample(
                                video_id=vid,
                                frame_id=frame_id,
                                image_path=img_path,
                                question=question,
                                answer=answer,
                            )
                        )

        if max_samples is not None:
            self.samples = self.samples[: max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        if self.processor is None:
            raise RuntimeError("Processor is required for SurgViVQA inference.")
        try:
            processed = self.processor(images=image, return_tensors="pt")
        except TypeError:
            processed = self.processor(image, return_tensors="pt")

        pixel_values = processed.get("pixel_values", None) if isinstance(processed, dict) else getattr(processed, "pixel_values", None)
        if pixel_values is None:
            raise RuntimeError("Processor output missing 'pixel_values'.")

        if self._log_processor_once:
            self._log_processor_once = False
            logging.info(
                "Processor: %s | pixel_values shape: %s",
                self.processor.__class__.__name__,
                tuple(pixel_values.shape),
            )

        return pixel_values.cpu().float()

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        pixel_values = self._process_image(image)
        frame = to_chw_frame(pixel_values)  # [3, H, W]

        # Repeat single frame to T frames to satisfy [T, 3, H, W]
        video_tensor = frame.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        return video_tensor, sample.question, sample.answer, sample.video_id, sample.frame_id, sample.image_path


def collate_ssgvqa(batch):
    videos, questions, answers, vids, frames, img_paths = [], [], [], [], [], []
    for v, q, a, vid, fid, ip in batch:
        videos.append(v)
        questions.append(q)
        answers.append(a)
        vids.append(vid)
        frames.append(fid)
        img_paths.append(ip)
    videos = torch.stack(videos, dim=0)  # [B, T, 3, H, W]
    return videos, questions, answers, vids, frames, img_paths


def build_logger(log_file: str) -> None:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_checkpoint(model, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info("[CKPT] loaded: %s", ckpt_path)
    logging.info("[CKPT] missing keys: %d", len(missing))
    logging.info("[CKPT] unexpected keys: %d", len(unexpected))

    # Enhancement 4: checkpoint load reliability
    if missing:
        logging.info("[CKPT] missing keys (first 30): %s", missing[:30])
    if unexpected:
        logging.info("[CKPT] unexpected keys (first 30): %s", unexpected[:30])

    total_params = sum(1 for _ in model.state_dict().keys())
    if len(unexpected) > 50 or (total_params > 0 and len(unexpected) / total_params > 0.1):
        logging.warning(
            "[CKPT] WARNING: %d unexpected keys (%.1f%% of model params). "
            "Checkpoint may not match current model structure. Evaluation results may be unreliable!",
            len(unexpected),
            100.0 * len(unexpected) / max(1, total_params),
        )


def evaluate(args) -> None:
    pred_dir = os.path.dirname(args.predictions_file)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- tokenizer / decoder ----
    decoder_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoder_model = AutoModelForCausalLM.from_pretrained(decoder_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )

    model = SurgViVQA(
        device=device,
        tokenizer=tokenizer,
        decoder_model=decoder_model,
        peft_config=lora_config,
        num_frames=args.num_frames,
    ).to(device)

    load_checkpoint(model, args.model_path)

    # Enhancement 5: num_frames consistency check
    model_num_frames = getattr(model, "num_frames", None)
    logging.info("[Config] args.num_frames=%d, model.num_frames=%s", args.num_frames, model_num_frames)
    if model_num_frames is not None and int(model_num_frames) != int(args.num_frames):
        logging.warning(
            "[Config] WARNING: num_frames mismatch! args.num_frames=%d but model.num_frames=%d. "
            "Consider using model.num_frames or adjusting --num-frames accordingly.",
            args.num_frames,
            model_num_frames,
        )

    dataset = SSGVQAStaticDataset(
        ssgvqa_root=args.ssgvqa_root,
        image_root=args.image_root,
        video_ids=args.videos,
        processor=model.processor,
        num_frames=args.num_frames,
        max_samples=args.max_samples,
        strict_missing_image=args.strict_missing_image,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_ssgvqa,
        pin_memory=True,
    )

    logging.info("Total samples: %d", len(dataset))

    total_samples = 0
    total_prompt_tokens = 0
    truncated_count = 0
    zero_label_tokens_count = 0
    all_label_tokens_scored: List[int] = []  # For diagnostics
    y_true: List[int] = []
    y_pred: List[int] = []
    anymatch_correct = 0
    anymatch_total = 0
    exact_map, norm_map = build_label_maps(SSGVQA_LABELS)

    logging.info("[Config] score_norm=%s", args.score_norm)

    with open(args.predictions_file, "w", encoding="utf-8") as fout:
        model.eval()
        with torch.no_grad():
            processed = 0
            for videos, questions, answers, vids, frames, img_paths in loader:
                if args.debug_first_n is not None and processed >= args.debug_first_n:
                    break

                if processed == 0:
                    logging.info("Images tensor shape: %s", tuple(videos.shape))

                prompt_prefixes = [build_prompt_prefix(q, args.prompt_mode) for q in questions]
                scores, tokenized_lens, truncated_flags, label_tokens_scored, label_token_lens = score_labels_closedset(
                    model=model,
                    tokenizer=tokenizer,
                    videos=videos,
                    prompt_prefixes=prompt_prefixes,
                    labels=SSGVQA_LABELS,
                    max_input_tokens=args.max_input_tokens,
                    device=device,
                    label_chunk_size=args.label_chunk_size,
                    score_norm=args.score_norm,
                )

                if (args.debug_first_n is not None) or (processed == 0):
                    head = prompt_prefixes[0][:120].replace("\n", "\\n")
                    tail = prompt_prefixes[0][-120:].replace("\n", "\\n")
                    logging.info(
                        "tokenized_len=%d | truncated=%s | max_input_tokens=%d",
                        int(tokenized_lens[0]),
                        bool(truncated_flags[0]),
                        int(args.max_input_tokens),
                    )
                    logging.info("prompt head: %s", head)
                    logging.info("prompt tail: %s", tail)

                if any(truncated_flags):
                    logging.warning("Truncation occurred in batch (max_input_tokens=%d).", int(args.max_input_tokens))

                for i, (q, gt, vid, fid, ip) in enumerate(
                    zip(questions, answers, vids, frames, img_paths)
                ):
                    gt_label = gt.strip()
                    gt_labels = [s.strip() for s in gt_label.split(",") if s.strip()]

                    # Build gt_set for any-match evaluation
                    gt_set = set()
                    gt_idx_first = -1
                    for candidate in gt_labels:
                        mapped_label, mapped_idx = map_pred_to_label(candidate, SSGVQA_LABELS, exact_map, norm_map)
                        if mapped_idx != -1:
                            gt_set.add(mapped_label)
                            if gt_idx_first == -1:
                                gt_idx_first = mapped_idx

                    row_scores = scores[i].detach().cpu()
                    pred_idx = int(torch.argmax(row_scores).item())
                    pred_label = SSGVQA_LABELS[pred_idx]

                    tokenized_len = int(tokenized_lens[i])
                    truncated = bool(truncated_flags[i])

                    # Enhancement 3: Truncation diagnostics
                    pred_label_tokens_scored = int(label_tokens_scored[i, pred_idx].item())
                    max_label_len_in_batch = max(label_token_lens)
                    estimated_concat_len = tokenized_len + max_label_len_in_batch

                    # Get label length for the predicted label
                    pred_label_len = label_token_lens[pred_idx]

                    total_samples += 1
                    total_prompt_tokens += tokenized_len
                    all_label_tokens_scored.append(pred_label_tokens_scored)
                    if truncated:
                        truncated_count += 1
                    if pred_label_tokens_scored == 0:
                        zero_label_tokens_count += 1
                        logging.warning(
                            "[WARN] Sample %d: label_tokens_scored=0 for pred_label=%s. "
                            "prefix_len=%d, label_len=%d, max_label_len_batch=%d, max_input_tokens=%d",
                            processed,
                            pred_label,
                            tokenized_len,
                            pred_label_len,
                            max_label_len_in_batch,
                            args.max_input_tokens,
                        )

                    # First-label metrics (single-label)
                    if gt_idx_first != -1:
                        y_true.append(gt_idx_first)
                        y_pred.append(pred_idx)

                    # Any-match metrics (multi-label)
                    if gt_set:
                        anymatch_total += 1
                        if pred_label in gt_set:
                            anymatch_correct += 1

                    topk = min(5, len(SSGVQA_LABELS))
                    top_scores, top_indices = torch.topk(row_scores, k=topk, largest=True)
                    top5 = [
                        {"label": SSGVQA_LABELS[int(idx)], "score": float(sc)}
                        for sc, idx in zip(top_scores.tolist(), top_indices.tolist())
                    ]

                    # Store JSONL record
                    fout.write(
                        json.dumps(
                            {
                                "video": vid,
                                "frame_id": fid,
                                "image_path": ip,
                                "question": q,
                                "gt_answer": gt_label,
                                "prompt_mode": args.prompt_mode,
                                "score_norm": args.score_norm,
                                "prompt_prefix": prompt_prefixes[i],
                                "prefix_token_len": tokenized_len,
                                "pred_label_len": pred_label_len,
                                "max_label_len_in_batch": max_label_len_in_batch,
                                "estimated_concat_len": estimated_concat_len,
                                "truncated": truncated,
                                "max_input_tokens": int(args.max_input_tokens),
                                "pred_label": pred_label,
                                "pred_idx": pred_idx,
                                "label_tokens_scored": pred_label_tokens_scored,
                                "top5": top5,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    if args.log_every_n > 0 and processed % args.log_every_n == 0:
                        logging.info("Sample q: %s", q)
                        logging.info("Sample gt: %s", gt_label)
                        logging.info("Sample pred label: %s", pred_label)
                        logging.info("Sample tokenized_len: %d", tokenized_len)
                        logging.info("Sample truncated: %s", truncated)

                    processed += 1
                    if args.debug_first_n is not None and processed >= args.debug_first_n:
                        break

    num_samples = int(total_samples)
    avg_prompt_token_len = float(total_prompt_tokens) / float(max(1, total_samples))
    truncated_ratio = float(truncated_count) / float(max(1, total_samples))
    zero_label_tokens_ratio = float(zero_label_tokens_count) / float(max(1, total_samples))
    anymatch_acc = float(anymatch_correct) / float(max(1, anymatch_total))

    # Compute label_tokens_scored stats
    if all_label_tokens_scored:
        avg_label_tokens_scored = float(np.mean(all_label_tokens_scored))
        min_label_tokens_scored = int(np.min(all_label_tokens_scored))
        max_label_tokens_scored = int(np.max(all_label_tokens_scored))
    else:
        avg_label_tokens_scored = 0.0
        min_label_tokens_scored = 0
        max_label_tokens_scored = 0

    metrics = compute_metrics(y_true, y_pred, num_classes=len(SSGVQA_LABELS))
    summary = {
        "num_samples": num_samples,
        "score_norm": args.score_norm,
        "avg_prompt_token_len": avg_prompt_token_len,
        "truncated_ratio": truncated_ratio,
        "zero_label_tokens_ratio": zero_label_tokens_ratio,
        "avg_label_tokens_scored": avg_label_tokens_scored,
        "min_label_tokens_scored": min_label_tokens_scored,
        "max_label_tokens_scored": max_label_tokens_scored,
        "acc_first_label": metrics["acc"],
        "mAP": metrics["mAP"],
        "mAR": metrics["mAR"],
        "mAF1": metrics["mAF1"],
        "wF1": metrics["wF1"],
        "anymatch_acc": anymatch_acc,
        "anymatch_total": anymatch_total,
        "anymatch_correct": anymatch_correct,
    }

    if zero_label_tokens_ratio > 0:
        logging.warning(
            "[Summary] %.2f%% samples have label_tokens_scored=0. "
            "Consider increasing --max-input-tokens (current=%d).",
            100.0 * zero_label_tokens_ratio,
            args.max_input_tokens,
        )

    logging.info("Summary: %s", summary)
    logging.info("Saved predictions: %s", args.predictions_file)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SurgViVQA evaluation on SSGVQA static QA")

    parser.add_argument("--model-path", required=True, help="Path to SurgViVQA checkpoint (.pth)")
    parser.add_argument("--ssgvqa-root", required=True, help="Root of SSGVQA static QA txt files")
    parser.add_argument("--image-root", required=True, help="Root of CholecT45 frames")
    parser.add_argument("--videos", nargs="+", required=True, help="Video IDs to evaluate")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)

    parser.add_argument("--prompt-mode", type=str, default="simple", choices=["simple", "choices"])
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--label-chunk-size", type=int, default=10)
    parser.add_argument("--score-norm", type=str, default="mean", choices=["sum", "mean"],
                        help="Score normalization: 'sum' for raw logprob sum, 'mean' for length-normalized (default)")

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strict-missing-image", action="store_true")

    parser.add_argument("--debug-first-n", type=int, default=None)
    parser.add_argument("--log-every-n", type=int, default=200)

    parser.add_argument("--log-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    build_logger(args.log_file)
    seed_everything(args.seed)
    evaluate(args)