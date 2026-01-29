import os
import json
import random
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import TaskType, LoraConfig

from dataloaders import STSGTemporalVideoQA, collate_stsg
from models.model import SurgViVQA
from utils.inference import inference
from utils.stsg_metrics import evaluate_stsg


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_args():
    p = argparse.ArgumentParser("SurgViVQA evaluation on STSG temporal QA")

    p.add_argument("--video_ids", nargs="+", required=True)
    p.add_argument("--stsg_qa_root", required=True)
    p.add_argument("--frame_root", required=True)
    p.add_argument("--checkpoint", required=True)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)

    # ---- frames ----
    p.add_argument("--num_frames", type=int, default=16)

    # ---- decoding controls ----
    p.add_argument("--max_prompt_len", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--decode_mode", type=str, default="closed", choices=["greedy", "hybrid", "closed"])

    # closed decoding options
    p.add_argument("--text_vocab_json", type=str, default=None,
                   help="Path to vocabs_train.json (category -> labels list). Required for closed decoding of text tasks.")
    p.add_argument("--count_max", type=int, default=60,
                   help="Closed decoding range for count: 0..count_max (set high enough to cover GT).")
    p.add_argument("--closed_text_topk", type=int, default=50,
                   help="For large text label spaces (e.g., ordering), shortlist before closed decoding.")

    # debug / sampling
    p.add_argument("--max_samples", type=int, default=None)

    # rationale (optional)
    p.add_argument("--include_rationale", action="store_true")
    p.add_argument("--rationale_key", type=str, default="auto")
    p.add_argument(
        "--append_rationale_to_question",
        action="store_true",
        help="Append selected rationale to the end of question text in dataloader."
    )

    # NEW: strict output format constraint in prompt
    p.add_argument("--strict_answer_format", action="store_true",
                   help="Append a strict output-format block right before FINAL_ANSWER: for all task types.")

    p.add_argument("--save_dir", type=str, default="stsg_eval_outputs_full")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded: {ckpt_path}")
    print(f"[CKPT] missing keys: {len(missing)}")
    print(f"[CKPT] unexpected keys: {len(unexpected)}")


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- tokenizer / decoder (LLM) ----
    decoder_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoder_model = AutoModelForCausalLM.from_pretrained(decoder_name)

    # LoRA config (inference_mode=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )

    # ---- model (PASS num_frames!) ----
    model = SurgViVQA(
        device=device,
        tokenizer=tokenizer,
        decoder_model=decoder_model,
        peft_config=lora_config,
        num_frames=args.num_frames,
    ).to(device)

    load_checkpoint(model, args.checkpoint)

    # ---- dataset / loader (PASS num_frames!) ----
    dataset = STSGTemporalVideoQA(
        video_ids=args.video_ids,
        stsg_qa_root=args.stsg_qa_root,
        frame_root=args.frame_root,
        num_frames=args.num_frames,
        processor=model.processor,
        max_samples=args.max_samples,
        strict_missing_frame=False,
        include_rationale=args.include_rationale,
        rationale_key=args.rationale_key,
        append_rationale_to_question=args.append_rationale_to_question,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_stsg,
        pin_memory=True,
    )

    # ---- inference ----
    references, predictions, metas = inference(
        val_loader=loader,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        decode_mode=args.decode_mode,
        vocabs_json=args.text_vocab_json,
        count_max=args.count_max,
        closed_text_topk=args.closed_text_topk,
        strict_answer_format=args.strict_answer_format,   # NEW
    )

    # ---- save per-sample jsonl ----
    pred_path = os.path.join(args.save_dir, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for gt, pr, m in zip(references, predictions, metas):
            f.write(json.dumps({
                "video_id": m.get("video_id"),
                "category": m.get("category"),
                "answer_type": m.get("answer_type"),
                "label_type": m.get("label_type"),
                "answer_mode": m.get("answer_mode"),
                "choice_K": m.get("choice_K"),
                "keyframes": m.get("keyframes"),
                "gt": gt,
                "pred_raw": pr,
            }, ensure_ascii=False) + "\n")
    print("Saved:", pred_path)

    # ---- metrics ----
    report = evaluate_stsg(references, predictions, metas, vocabs_json=args.text_vocab_json)
    report_path = os.path.join(args.save_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Saved:", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    main(args)
