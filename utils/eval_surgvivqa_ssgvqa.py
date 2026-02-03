"""
SurgViVQA evaluation on SSGVQA static QA.

Key goals:
- Always log full raw generation text.
- Ensure images tensor shape is [B, T, 3, H, W].
- Support prompt modes: simple vs choices.
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


def build_prompt(question: str, mode: str) -> str:
    q = (question or "").strip()
    if mode == "choices":
        candidates = ", ".join(SSGVQA_LABELS)
        return (
            "You are given a surgical image. Choose the best answer from the candidate list.\n"
            f"Candidates: {candidates}\n"
            f"Question: {q}\n"
            "Answer:"
        )
    return (
        "You are given a surgical image. Answer the question based on the image. Answer concisely.\n"
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


def _get_original_lengths(tokenizer, prompts: List[str]) -> List[int]:
    try:
        raw = tokenizer(prompts, padding=False, truncation=False, return_length=True)
        if "length" in raw:
            return [int(x) for x in raw["length"]]
    except TypeError:
        pass
    lengths = []
    for p in prompts:
        enc = tokenizer(p, padding=False, truncation=False)
        lengths.append(len(enc["input_ids"]))
    return lengths


def generate_outputs(
    model,
    tokenizer,
    videos: torch.Tensor,
    prompts: List[str],
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_beams: int,
    device,
):
    tok = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)

    video_embeds, video_atts = model.encode_video(videos.to(device))
    text_out = model.text_encoder(
        input_ids=input_ids,
        attention_mask=attn_mask,
        encoder_hidden_states=video_embeds,
        encoder_attention_mask=video_atts,
        return_dict=True,
    )
    text_embeds = text_out.last_hidden_state

    gen_ids = model.llm.generate(
        inputs_embeds=text_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    max_prompt_len = input_ids.size(1)
    gen_only = gen_ids[:, max_prompt_len:]

    outputs: List[str] = []
    output_token_lens: List[int] = []
    for i in range(gen_only.size(0)):
        seq = gen_only[i].tolist()
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in seq:
            seq = seq[: seq.index(tokenizer.eos_token_id)]
        output_token_lens.append(len(seq))
        outputs.append(tokenizer.decode(seq, skip_special_tokens=True).strip())

    return tok, outputs, output_token_lens


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
    empty_output_count = 0
    total_output_tokens = 0
    total_prompt_tokens = 0
    truncated_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    exact_map, norm_map = build_label_maps(SSGVQA_LABELS)

    with open(args.predictions_file, "w", encoding="utf-8") as fout:
        model.eval()
        with torch.no_grad():
            processed = 0
            for videos, questions, answers, vids, frames, img_paths in loader:
                if args.debug_first_n is not None and processed >= args.debug_first_n:
                    break

                if processed == 0:
                    logging.info("Images tensor shape: %s", tuple(videos.shape))

                prompts = [build_prompt(q, args.prompt_mode) for q in questions]
                orig_lens = _get_original_lengths(tokenizer, prompts)
                tok, raw_preds, output_token_lens = generate_outputs(
                    model=model,
                    tokenizer=tokenizer,
                    videos=videos,
                    prompts=prompts,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    device=device,
                )

                tokenized_lens = tok["attention_mask"].sum(dim=1).tolist()
                truncated_flags = [int(orig_lens[i] > args.max_input_tokens) for i in range(len(prompts))]

                if (args.debug_first_n is not None) or (processed == 0):
                    head = prompts[0][:120].replace("\n", "\\n")
                    tail = prompts[0][-120:].replace("\n", "\\n")
                    logging.info(
                        "tokenized_len=%d | truncated=%s | max_input_tokens=%d",
                        int(tokenized_lens[0]),
                        bool(truncated_flags[0]),
                        int(args.max_input_tokens),
                    )
                    logging.info("prompt head: %s", head)
                    logging.info("prompt tail: %s", tail)

                for i, (q, gt, vid, fid, ip, pred_raw) in enumerate(
                    zip(questions, answers, vids, frames, img_paths, raw_preds)
                ):
                    gt_label = gt.strip()
                    gt_labels = [s.strip() for s in gt_label.split(",") if s.strip()]
                    gt_idx = -1
                    if gt_labels:
                        for candidate in gt_labels:
                            mapped_label, mapped_idx = map_pred_to_label(candidate, SSGVQA_LABELS, exact_map, norm_map)
                            if mapped_idx != -1:
                                gt_idx = mapped_idx
                                break

                    pred_label, pred_idx = map_pred_to_label(pred_raw, SSGVQA_LABELS, exact_map, norm_map)
                    tokenized_len = int(tokenized_lens[i])
                    truncated = bool(truncated_flags[i])
                    output_token_len = int(output_token_lens[i])

                    total_samples += 1
                    total_output_tokens += output_token_len
                    total_prompt_tokens += tokenized_len
                    if truncated:
                        truncated_count += 1
                    if not pred_raw:
                        empty_output_count += 1
                    if gt_idx != -1 and pred_idx != -1:
                        y_true.append(gt_idx)
                        y_pred.append(pred_idx)

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
                                "prompt_used": prompts[i],
                                "raw_pred_text": pred_raw,
                                "tokenized_len": tokenized_len,
                                "truncated": truncated,
                                "max_input_tokens": int(args.max_input_tokens),
                                "max_new_tokens": int(args.max_new_tokens),
                                "output_token_len": output_token_len,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    if args.log_every_n > 0 and processed % args.log_every_n == 0:
                        logging.info("Sample q: %s", q)
                        logging.info("Sample gt: %s", gt_label)
                        logging.info("Sample pred raw: %s", pred_raw)
                        logging.info("Sample tokenized_len: %d", tokenized_len)
                        logging.info("Sample output_token_len: %d", output_token_len)

                    processed += 1
                    if args.debug_first_n is not None and processed >= args.debug_first_n:
                        break

    num_samples = int(total_samples)
    empty_output_ratio = float(empty_output_count) / float(max(1, total_samples))
    avg_output_token_len = float(total_output_tokens) / float(max(1, total_samples))
    avg_prompt_token_len = float(total_prompt_tokens) / float(max(1, total_samples))
    truncated_ratio = float(truncated_count) / float(max(1, total_samples))

    metrics = compute_metrics(y_true, y_pred, num_classes=len(SSGVQA_LABELS))
    summary = {
        "num_samples": num_samples,
        "empty_output_ratio": empty_output_ratio,
        "avg_output_token_len": avg_output_token_len,
        "avg_prompt_token_len": avg_prompt_token_len,
        "truncated_ratio": truncated_ratio,
        "acc": metrics["acc"],
        "mAP": metrics["mAP"],
        "mAR": metrics["mAR"],
        "mAF1": metrics["mAF1"],
        "wF1": metrics["wF1"],
    }

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
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)

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