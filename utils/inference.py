import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


def _parse_choice_K_from_question(q: str) -> Optional[int]:
    if not q:
        return None
    qq = q.strip().lower()
    m = re.search(r"(answer|reply)\s*(with\s*(a\s*)?number\s*)?1\s*(\-|–|to)\s*(\d+)\b", qq)
    if m:
        try:
            return int(m.group(5))
        except Exception:
            return None
    return None


def _parse_bool_pred(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.strip().lower()
    if re.search(r"\b(yes|true)\b", t):
        return 1
    if re.search(r"\b(no|false)\b", t):
        return 0
    m = re.search(r"\b([01])\b", t)
    if m:
        return int(m.group(1))
    return None


def _infer_label_type_from_meta(question: str, meta: Dict[str, Any]) -> str:
    lt = (meta.get("label_type") or "").lower().strip()
    if not lt:
        lt = (meta.get("answer_mode") or "").lower().strip()
    if not lt:
        lt = (meta.get("answer_type") or "").lower().strip()

    if lt in ("binary", "boolean", "bin", "yesno", "yes/no"):
        lt = "bool"
    if lt in ("choice", "mcq", "multi_choice", "multichoice", "option", "index"):
        lt = "choice_index"
    if lt in ("numeric", "number", "float", "int", "reg"):
        lt = "seconds"
    if lt in ("cat", "text", "string", "phrase"):
        lt = "text"

    # question cue
    if _parse_choice_K_from_question(question) is not None:
        return "choice_index"

    # category cues
    cat = (meta.get("category") or "").lower().strip()
    if cat == "count":
        return "count"
    if cat == "duration":
        return "duration"
    if cat == "motion":
        return "text"

    return lt or "text"


def _load_closed_labels(vocabs_json: str) -> Dict[str, List[str]]:
    with open(vocabs_json, "r", encoding="utf-8") as f:
        voc = json.load(f)
    out: Dict[str, List[str]] = {}
    for k, arr in voc.items():
        if not isinstance(arr, list):
            continue
        kk = str(k).lower()
        out[kk] = [str(x) for x in arr]
    return out


def _labels_for_category(closed_labels: Dict[str, List[str]], cat: str) -> List[str]:
    """
    Use vocab labels as closed candidates.
    By default, drop <UNK> to avoid degenerate selection, except for motion where <UNK> is explicitly defined.
    """
    labels = closed_labels.get(cat, [])
    if not labels:
        return labels
    if cat == "motion":
        return labels
    return [x for x in labels if x.strip() != "<UNK>"]


def _build_strict_format_block(
    question: str,
    meta: Dict[str, Any],
    label_type: str,
    cat: str,
    K: Optional[int],
    choices: Any,
    closed_labels: Optional[Dict[str, List[str]]],
    count_max: int,
    closed_text_topk: int,
) -> str:
    """
    Build a very strict output-format instruction block.
    This block should be placed RIGHT BEFORE 'FINAL_ANSWER:' (end of prompt).
    """
    lines: List[str] = []
    lines.append("### OUTPUT FORMAT (STRICT)")
    lines.append("Return EXACTLY ONE LINE with ONLY the answer. No explanation, no extra words.")
    lines.append("Do NOT use quotes. Do NOT add punctuation at the end.")
    lines.append("If unsure, guess a valid answer in the required format.")

    # Normalize K
    if K is None:
        K = meta.get("choice_K", None)
    if K is None:
        K = _parse_choice_K_from_question(question)

    if label_type == "bool":
        # Keep aligned with your parsing/closed decoding (YES/NO)
        lines.append("Answer must be exactly: YES or NO (uppercase).")
        lines.append("Regex: ^(YES|NO)$")

    elif label_type == "count":
        # Optionally keep within [0..count_max] to reduce out-of-range
        lines.append(f"Answer must be an integer in [0, {int(count_max)}]. Digits only.")
        lines.append("Regex: ^[0-9]+$")

    elif label_type in ("seconds", "duration"):
        # A single float or int; no units
        lines.append("Answer must be a single number (integer or decimal). No units.")
        lines.append("Regex: ^[0-9]+(\\.[0-9]+)?$")

    elif label_type == "choice_index" or (isinstance(K, int) and K > 0) or (choices is not None):
        if isinstance(K, int) and K > 0:
            lines.append(f"Answer must be an integer option number from 1 to {K}. Digits only.")
        else:
            lines.append("Answer must be an integer option number (1,2,3,...). Digits only.")
        lines.append("Regex: ^[0-9]+$")

    else:
        # text-like tasks
        if cat == "motion":
            lines.append(
                "Answer must be EXACTLY one label from: "
                "<UNK>, decrease, downward, increase, leftward, nearly unchanged, rightward, stationary, upward."
            )
        elif closed_labels is not None and cat in closed_labels:
            # If label space not too big, list it for stronger constraint
            lbls = _labels_for_category(closed_labels, cat)
            # Use shortlist for very large spaces (still too big to print)
            if len(lbls) <= 60:
                lines.append("Answer must be EXACTLY one label from the following list:")
                lines.append(", ".join(lbls))
            else:
                # For large spaces (ordering/extreme/phase etc), avoid dumping all labels
                # Provide structural constraint instead (still strict format)
                if cat == "ordering":
                    lines.append("Answer must be a SINGLE label phrase, OR strictly in the form: <event1> before <event2>.")
                    lines.append("Do NOT add any other words besides the label content itself.")
                else:
                    lines.append("Answer must be a SINGLE short label phrase (no explanation).")
                    lines.append("Use the same wording style as labels (lowercase words / underscores if present).")
        else:
            if cat == "ordering":
                lines.append("Answer must be a SINGLE label phrase, OR strictly in the form: <event1> before <event2>.")
            else:
                lines.append("Answer must be a SINGLE short phrase label (no explanation).")

    # Put the final reminder right before FINAL_ANSWER:
    lines.append("Write ONLY the answer after 'FINAL_ANSWER:'")
    return "\n".join(lines)


def _build_prompt(
    question: str,
    meta: Dict[str, Any],
    closed_labels: Optional[Dict[str, List[str]]] = None,
    strict_answer_format: bool = False,
    count_max: int = 60,
    closed_text_topk: int = 50,
) -> str:
    q = _clean_text(question)

    label_type = _infer_label_type_from_meta(q, meta)
    cat = (meta.get("category") or "").lower().strip()
    if cat == "phase":
        cat = "phase_transition"

    K = meta.get("choice_K", None)
    if K is None:
        K = _parse_choice_K_from_question(q)

    choices = meta.get("choices", None)

    # ---- (old) light instruction, keep for non-strict mode ----
    if label_type == "bool":
        instr = "Answer with only YES or NO."
    elif label_type in ("seconds", "duration"):
        instr = "Answer with only a number (can be float)."
    elif label_type == "count":
        instr = "Answer with only an integer number."
    elif label_type == "choice_index" or (choices is not None) or (isinstance(K, int) and K > 0):
        if isinstance(K, int) and K > 0:
            instr = f"Answer with only a number 1-{K}."
        else:
            instr = "Select the best option and answer with only the option number (1,2,3,...)."
    else:
        if closed_labels is not None and cat in closed_labels:
            lbls = _labels_for_category(closed_labels, cat)
            if len(lbls) <= 60:
                instr = "Answer with exactly one label from: " + ", ".join(lbls) + "."
            else:
                instr = "Answer with a short phrase."
        elif cat == "motion":
            instr = (
                "Answer with exactly one label from: "
                "<UNK>, decrease, downward, increase, leftward, nearly unchanged, rightward, stationary, upward."
            )
        else:
            instr = "Answer with a short phrase."

    # ---- options block (choice questions) ----
    opt_block = ""
    if (label_type == "choice_index" or choices is not None) and isinstance(choices, list) and len(choices) > 0:
        lines = [f"{i}) {str(c)}" for i, c in enumerate(choices, start=1)]
        opt_block = "\nOptions:\n" + "\n".join(lines)

    # ---- rationale block (avoid duplication if already appended into question) ----
    rat = meta.get("rationale", None)
    rat_block = ""
    if not bool(meta.get("rationale_appended", False)):
        if isinstance(rat, str) and rat.strip():
            rat_block = "\nRationale:\n" + rat.strip()

    # ---- strict format block at the very end ----
    strict_block = ""
    if strict_answer_format:
        strict_block = "\n" + _build_strict_format_block(
            question=q,
            meta=meta,
            label_type=label_type,
            cat=cat,
            K=K if isinstance(K, int) else None,
            choices=choices,
            closed_labels=closed_labels,
            count_max=count_max,
            closed_text_topk=closed_text_topk,
        )

    prompt = (
        f"Question: {q}"
        f"{opt_block}"
        f"{rat_block}\n"
    )

    if not strict_answer_format:
        prompt += f"{instr}\n"

    # IMPORTANT: strict block must be placed right before FINAL_ANSWER:
    prompt += f"{strict_block}\nFINAL_ANSWER:"
    return prompt


def _encode_video_cached(model, images: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = images.to(device)
    if hasattr(model, "encode_video"):
        with torch.no_grad():
            return model.encode_video(images)
    with torch.no_grad():
        video_embeds = model.visual_encoder(pixel_values=images).last_hidden_state
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long, device=device)
    return video_embeds, video_atts


def _forward_logits(
    model,
    qa_input_ids: torch.Tensor,
    qa_att_mask: torch.Tensor,
    video_embeds: torch.Tensor,
    video_atts: torch.Tensor,
) -> torch.Tensor:
    return model(
        video=None,
        qa_inputs_ids=qa_input_ids,
        qa_att_mask=qa_att_mask,
        video_embeds=video_embeds,
        video_atts=video_atts,
    )


def _score_completions_logprob_batched(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,          # (Li,)  !!! MUST be true length, no padding
    video_embeds_1: torch.Tensor,      # (1, V, D)
    video_atts_1: torch.Tensor,        # (1, V)
    completions: List[str],
    device,
    batch_size: int = 32,
) -> List[float]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_len = int(prompt_ids.numel())

    comp_ids_all: List[List[int]] = [tokenizer.encode(c, add_special_tokens=False) for c in completions]
    scores: List[float] = []

    for s in range(0, len(completions), batch_size):
        chunk_ids = comp_ids_all[s : s + batch_size]
        bsz = len(chunk_ids)
        max_comp = max((len(x) for x in chunk_ids), default=0)
        if max_comp == 0:
            scores.extend([-1e9] * bsz)
            continue

        total_len = prompt_len + max_comp
        ids = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((bsz, total_len), dtype=torch.long, device=device)

        # prompt (all valid)
        ids[:, :prompt_len] = prompt_ids.unsqueeze(0).expand(bsz, -1)
        att[:, :prompt_len] = 1

        # completion (variable valid)
        for j, toks in enumerate(chunk_ids):
            L = len(toks)
            if L == 0:
                continue
            ids[j, prompt_len : prompt_len + L] = torch.tensor(toks, dtype=torch.long, device=device)
            att[j, prompt_len : prompt_len + L] = 1

        vemb = video_embeds_1.expand(bsz, -1, -1).contiguous()
        vats = video_atts_1.expand(bsz, -1).contiguous()

        logits = _forward_logits(model, ids, att, vemb, vats)  # (B, total_len, vocab)
        log_soft = torch.log_softmax(logits, dim=-1)

        for j, toks in enumerate(chunk_ids):
            lp = 0.0
            ok = True
            for k, tok_id in enumerate(toks):
                pos = prompt_len + k - 1
                if pos < 0 or pos >= log_soft.size(1):
                    ok = False
                    break
                lp += float(log_soft[j, pos, tok_id].item())
            scores.append(lp if ok else -1e9)

    return scores


def _closed_predict_bool(model, tokenizer, prompt_ids, vemb, vats, device) -> str:
    """
    More robust than only YES/NO:
      positive candidates: YES / TRUE / 1
      negative candidates: NO / FALSE / 0
    We take the best-scoring candidate in each group and compare.
    """
    pos = [" YES", " True", " TRUE", " 1"]
    neg = [" NO", " False", " FALSE", " 0"]
    cands = pos + neg
    scores = _score_completions_logprob_batched(model, tokenizer, prompt_ids, vemb, vats, cands, device, batch_size=len(cands))

    pos_best = max(scores[:len(pos)])
    neg_best = max(scores[len(pos):])

    return "YES" if pos_best >= neg_best else "NO"


def _closed_predict_choice(model, tokenizer, prompt_ids, vemb, vats, K: int, device) -> str:
    completions = [f" {i}" for i in range(1, K + 1)]
    scores = _score_completions_logprob_batched(model, tokenizer, prompt_ids, vemb, vats, completions, device, batch_size=min(32, K))
    best = int(torch.tensor(scores).argmax().item())
    return str(best + 1)


def _closed_predict_count(model, tokenizer, prompt_ids, vemb, vats, count_max: int, device) -> str:
    completions = [f" {i}" for i in range(0, count_max + 1)]
    bs = 64 if (count_max + 1) >= 64 else (count_max + 1)
    scores = _score_completions_logprob_batched(model, tokenizer, prompt_ids, vemb, vats, completions, device, batch_size=bs)
    best = int(torch.tensor(scores).argmax().item())
    return str(best)


def _shortlist_labels_by_overlap(labels: List[str], question: str, topk: int = 50) -> List[str]:
    if topk <= 0 or len(labels) <= topk:
        return labels

    q = (question or "").lower()
    q_toks = set(re.findall(r"[a-z0-9_]+", q))

    scored = []
    for lab in labels:
        l = lab.lower()
        l_toks = set(re.findall(r"[a-z0-9_]+", l))
        inter = len(q_toks & l_toks)
        bonus = 0
        if ("before" in q or "after" in q) and ("before" in l or "after" in l):
            bonus = 1
        scored.append((inter + bonus, lab))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [lab for _, lab in scored[:topk]]


def _closed_predict_vocab_text(model, tokenizer, prompt_ids, vemb, vats, labels: List[str], device) -> str:
    completions = [(" " + s) if not s.startswith(" ") else s for s in labels]
    scores = _score_completions_logprob_batched(model, tokenizer, prompt_ids, vemb, vats, completions, device, batch_size=32)
    best = int(torch.tensor(scores).argmax().item())
    return labels[best]


def batch_decode(
    images: torch.Tensor,
    questions: List[str],
    metas: List[Dict[str, Any]],
    model,
    tokenizer,
    max_prompt_len: int,
    max_new_tokens: int,
    device,
    decode_mode: str = "hybrid",
    closed_labels: Optional[Dict[str, List[str]]] = None,
    count_max: int = 20,
    closed_text_topk: int = 50,
    strict_answer_format: bool = False,
) -> List[str]:
    """
    decode_mode:
      - greedy: pure greedy generation
      - closed: ALWAYS closed-decoding for bool/choice/count AND for text categories that exist in vocabs_json
      - hybrid: greedy first; if output is not parseable / not in label space, fallback to closed-decoding
    """
    bsz = images.size(0)

    prompts = [
        _build_prompt(
            q, m,
            closed_labels=closed_labels,
            strict_answer_format=strict_answer_format,
            count_max=int(count_max),
            closed_text_topk=int(closed_text_topk),
        )
        for q, m in zip(questions, metas)
    ]

    tok = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)

    video_embeds, video_atts = _encode_video_cached(model, images, device)

    total_len = max_prompt_len + max_new_tokens
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_ids = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    padded_att = torch.zeros((bsz, total_len), dtype=torch.long, device=device)

    L0 = input_ids.size(1)
    padded_ids[:, :L0] = input_ids
    padded_att[:, :L0] = attn_mask

    valid_lens = padded_att.sum(dim=1).long()
    batch_idx = torch.arange(bsz, device=device)

    only_new = torch.empty((bsz, 0), dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            max_valid = int(valid_lens.max().item())
            if max_valid >= total_len:
                break

            ids_now = padded_ids[:, :max_valid]
            att_now = padded_att[:, :max_valid]

            logits = _forward_logits(model, ids_now, att_now, video_embeds, video_atts)
            next_logits = logits[batch_idx, valid_lens - 1]
            next_ids = torch.argmax(next_logits, dim=-1)

            if tokenizer.eos_token_id is not None:
                next_ids = torch.where(finished, torch.tensor(pad_id, device=device), next_ids)
                finished = finished | (next_ids == tokenizer.eos_token_id)

            padded_ids[batch_idx, valid_lens] = next_ids
            padded_att[batch_idx, valid_lens] = (~finished).long()
            valid_lens = valid_lens + (~finished).long()

            only_new = torch.cat([only_new, next_ids.unsqueeze(1)], dim=1)
            if bool(finished.all()):
                break

    greedy_texts: List[str] = []
    for i in range(bsz):
        seq = only_new[i]
        toks = seq.tolist()
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in toks:
            toks = toks[: toks.index(tokenizer.eos_token_id)]
        greedy_texts.append(tokenizer.decode(toks, skip_special_tokens=True).strip())

    outs: List[str] = []
    for i in range(bsz):
        meta = metas[i]
        q = questions[i]
        pred = greedy_texts[i]

        lt = _infer_label_type_from_meta(q, meta)

        cat = (meta.get("category") or "").lower().strip()
        if cat == "phase":
            cat = "phase_transition"

        K = meta.get("choice_K", None)
        if K is None:
            K = _parse_choice_K_from_question(q)
        choices = meta.get("choices", None)
        if K is None and isinstance(choices, list):
            K = len(choices)

        # per-sample true prompt length (no padding)
        Li = int(attn_mask[i].sum().item())
        prompt_ids_1 = input_ids[i, :Li].detach()

        vemb = video_embeds[i : i + 1]
        vats = video_atts[i : i + 1]

        # ---- BOOL ----
        if lt == "bool":
            if decode_mode == "closed":
                outs.append(_closed_predict_bool(model, tokenizer, prompt_ids_1, vemb, vats, device))
            else:
                pb = _parse_bool_pred(pred)
                if pb is None:
                    outs.append(_closed_predict_bool(model, tokenizer, prompt_ids_1, vemb, vats, device))
                else:
                    outs.append(pred)
            continue

        # ---- CHOICE ----
        if lt == "choice_index" or (isinstance(K, int) and K > 0) or (choices is not None):
            if not isinstance(K, int) or K <= 0:
                outs.append(pred)
                continue
            if decode_mode == "closed":
                outs.append(_closed_predict_choice(model, tokenizer, prompt_ids_1, vemb, vats, int(K), device))
            else:
                m = re.search(r"\b(\d+)\b", pred)
                if m is None:
                    outs.append(_closed_predict_choice(model, tokenizer, prompt_ids_1, vemb, vats, int(K), device))
                else:
                    pi = int(m.group(1))
                    if 1 <= pi <= K:
                        outs.append(pred)
                    else:
                        outs.append(_closed_predict_choice(model, tokenizer, prompt_ids_1, vemb, vats, int(K), device))
            continue

        # ---- COUNT ----
        if lt == "count":
            if decode_mode == "closed":
                outs.append(_closed_predict_count(model, tokenizer, prompt_ids_1, vemb, vats, int(count_max), device))
            else:
                m = re.search(r"\b(\d+)\b", pred)
                if m is None:
                    outs.append(_closed_predict_count(model, tokenizer, prompt_ids_1, vemb, vats, int(count_max), device))
                else:
                    outs.append(pred)
            continue

        # ---- TEXT (closed-set) ----
        if closed_labels is not None and cat in closed_labels:
            labels_full = _labels_for_category(closed_labels, cat)
            labels = labels_full if labels_full else closed_labels[cat]

            if len(labels) > max(60, closed_text_topk):
                labels_use = _shortlist_labels_by_overlap(labels, q, topk=int(closed_text_topk))
            else:
                labels_use = labels

            if decode_mode == "closed":
                outs.append(_closed_predict_vocab_text(model, tokenizer, prompt_ids_1, vemb, vats, labels_use, device))
            else:
                low = pred.strip().lower()
                if any(low == lab.strip().lower() for lab in labels_use):
                    outs.append(pred)
                else:
                    outs.append(_closed_predict_vocab_text(model, tokenizer, prompt_ids_1, vemb, vats, labels_use, device))
            continue

        outs.append(pred)

    return outs


def inference(
    val_loader,
    model,
    tokenizer,
    device,
    max_prompt_len: int,
    max_new_tokens: int,
    decode_mode: str = "hybrid",
    vocabs_json: Optional[str] = None,
    count_max: int = 20,
    closed_text_topk: int = 50,
    strict_answer_format: bool = False,
):
    references: List[Any] = []
    predictions: List[str] = []
    metas: List[Dict[str, Any]] = []

    closed_labels = _load_closed_labels(vocabs_json) if vocabs_json else None

    model.eval()
    with torch.no_grad():
        for _, (images, questions, answers, batch_metas) in enumerate(tqdm(val_loader), 0):
            images = images.to(device)
            gen = batch_decode(
                images=images,
                questions=questions,
                metas=batch_metas,
                model=model,
                tokenizer=tokenizer,
                max_prompt_len=max_prompt_len,
                max_new_tokens=max_new_tokens,
                device=device,
                decode_mode=decode_mode,
                closed_labels=closed_labels,
                count_max=count_max,
                closed_text_topk=closed_text_topk,
                strict_answer_format=strict_answer_format,
            )
            references.extend(answers)
            predictions.extend(gen)
            metas.extend(batch_metas)

    print("First 5 GT:", references[:5])
    print("First 5 Pred:", predictions[:5])
    print("First 5 Meta:", metas[:5])
    return references, predictions, metas
