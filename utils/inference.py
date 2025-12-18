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


def _parse_choice_K_from_question(question: str) -> Optional[int]:
    """
    Detect choice questions from templates like:
      - "Answer with a number 1-3"
      - "Reply 1-5"
      - "Answer 1-4"
    Return K if found, else None.
    """
    if not question:
        return None
    q = question.lower()
    m = re.search(r"\b(?:answer|reply|respond)\b[^0-9]{0,40}\b1\s*[-–]\s*(\d+)\b", q)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\b1\s*[-–]\s*(\d+)\b", q)
    if m and ("answer" in q or "reply" in q or "respond" in q):
        try:
            return int(m.group(1))
        except Exception:
            return None
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


def _build_ordering_event_set(ordering_labels: List[str]) -> set:
    """Extract event phrases from ordering labels of form '<e1> before <e2>'"""
    ev = set()
    for lab in ordering_labels:
        if not isinstance(lab, str):
            continue
        s = lab.strip()
        if " before " in s:
            a, b = s.split(" before ", 1)
            a = a.strip()
            b = b.strip()
            if a and b:
                ev.add(a)
                ev.add(b)
    return ev


def _load_closed_labels(vocabs_json: str) -> Dict[str, List[str]]:
    """
    Load per-category vocab labels (closed-set candidates) from vocabs_train.json.

    Additionally, derive a set of *event phrases* from ordering labels of the form:
        '<event1> before <event2>'

    This supports the 'structured candidates' strategy for ordering_text:
      - Extract two events from the question
      - Only score 2 candidates: 'e1 before e2' vs 'e2 before e1'
    """
    with open(vocabs_json, "r", encoding="utf-8") as f:
        voc = json.load(f)

    out: Dict[str, List[str]] = {}
    for k, arr in voc.items():
        if not isinstance(arr, list):
            continue
        kk = str(k).lower().strip()
        out[kk] = [str(x) for x in arr]

    # Special auxiliary resources (not a real category)
    ordering_labels = out.get("ordering", [])
    if ordering_labels:
        ordering_events = _build_ordering_event_set(ordering_labels)
        out["__ordering_events__"] = sorted(ordering_events)

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


def _build_prompt(question: str, meta: Dict[str, Any], closed_labels: Optional[Dict[str, List[str]]] = None) -> str:
    q = _clean_text(question)

    label_type = _infer_label_type_from_meta(q, meta)
    cat = (meta.get("category") or "").lower().strip()
    if cat == "phase":
        cat = "phase_transition"

    K = meta.get("choice_K", None)
    if K is None:
        K = _parse_choice_K_from_question(q)

    choices = meta.get("choices", None)

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

            # For small closed sets (e.g., motion), listing labels can help.
            # For large label spaces (especially ordering), do NOT list labels to avoid prompt truncation.
            if len(lbls) <= 60 and cat not in ("ordering",):
                instr = "Answer with exactly one label from: " + ", ".join(lbls) + "."
            else:
                if cat == "ordering":
                    instr = (
                        "Answer with ONLY the final label/phrase. "
                        "If your answer expresses temporal order, use the exact format: '<event1> before <event2>'. "
                        "Do NOT use the word 'after'. Do NOT add explanation or punctuation."
                    )
                else:
                    instr = "Answer with a short phrase."
        elif cat == "motion":
            instr = (
                "Answer with exactly one label from: "
                "<UNK>, decrease, downward, increase, leftward, nearly unchanged, rightward, stationary, upward."
            )
        else:
            instr = "Answer with a short phrase."

    opt_block = ""
    if (label_type == "choice_index" or choices is not None) and isinstance(choices, list) and len(choices) > 0:
        lines = [f"{i}) {str(c)}" for i, c in enumerate(choices, start=1)]
        opt_block = "\nOptions:\n" + "\n".join(lines)

    rat = meta.get("rationale", None)
    rat_block = ""
    if isinstance(rat, str) and rat.strip():
        rat_block = "\nContext:\n" + rat.strip()

    prompt = (
        "You are an expert surgical video QA assistant.\n"
        f"Question:\n{q}"
        f"{opt_block}"
        f"{rat_block}\n"
        f"{instr}\n"
        "FINAL_ANSWER:"
    )
    return prompt


def _forward_logits(model, input_ids, attention_mask, video_embeds, video_atts, video=None):
    """
    SurgViVQA forward signature (from your model.py):
      forward(video, qa_inputs_ids, qa_att_mask, video_embeds=None, video_atts=None)
    Return: logits tensor [B, L, vocab]
    """
    # Prefer calling SurgViVQA-style forward
    try:
        out = model(
            video,                    # can be None if video_embeds/video_atts are provided
            input_ids,
            attention_mask,
            video_embeds=video_embeds,
            video_atts=video_atts,
        )
    except TypeError:
        # Fallback: try keyword names that match SurgViVQA.forward
        out = model(
            video=video,
            qa_inputs_ids=input_ids,
            qa_att_mask=attention_mask,
            video_embeds=video_embeds,
            video_atts=video_atts,
        )

    # Some models return dict/obj; SurgViVQA returns logits directly (Tensor)
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    if hasattr(out, "logits"):
        return out.logits
    return out


def _score_completions_logprob_batched(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,          # (Li,)  !!! MUST be true length, no padding
    video_embeds_1: torch.Tensor,      # (1, V, D)
    video_atts_1: torch.Tensor,        # (1, V)
    completions: List[str],
    device,
    batch_size: int = 32,
    length_norm_alpha: float = 1.0,    # 1.0 ~= mean logprob; 0.0 ~= sum logprob
) -> List[float]:
    """
    Score each completion y by:
        score(y) = sum_{k} log p(tok_k | prompt, tok_<k, video) / (len(y) ** alpha)

    This reduces the strong bias toward short labels when doing closed-set decoding.
    """
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

        comp_lens = [0] * bsz
        # completion (variable valid)
        for j, toks in enumerate(chunk_ids):
            L = len(toks)
            comp_lens[j] = L
            if L == 0:
                continue
            ids[j, prompt_len : prompt_len + L] = torch.tensor(toks, dtype=torch.long, device=device)
            att[j, prompt_len : prompt_len + L] = 1

        vemb = video_embeds_1.expand(bsz, -1, -1).contiguous()
        vats = video_atts_1.expand(bsz, -1).contiguous()

        logits = _forward_logits(model, ids, att, vemb, vats, video=None)  # (B, total_len, vocab)
        log_soft = torch.log_softmax(logits, dim=-1)

        for j, toks in enumerate(chunk_ids):
            L = comp_lens[j]
            if L <= 0:
                scores.append(-1e9)
                continue

            lp_sum = 0.0
            ok = True
            for k, tok_id in enumerate(toks):
                pos = prompt_len + k - 1
                if pos < 0 or pos >= log_soft.size(1):
                    ok = False
                    break
                lp_sum += float(log_soft[j, pos, tok_id].item())

            if not ok:
                scores.append(-1e9)
                continue

            denom = (float(L) ** float(length_norm_alpha)) if length_norm_alpha and length_norm_alpha > 0 else 1.0
            scores.append(lp_sum / denom)

    return scores


def _closed_predict_bool(model, tokenizer, prompt_ids, vemb, vats, device) -> str:
    pos = [" YES", " Yes", " TRUE", " True", " 1"]
    neg = [" NO", " No", " FALSE", " False", " 0"]
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
    scores = _score_completions_logprob_batched(model, tokenizer, prompt_ids, vemb, vats, completions, device, batch_size=min(64, len(completions)))
    best = int(torch.tensor(scores).argmax().item())
    return str(best)


def _closed_predict_vocab_text(model, tokenizer, prompt_ids, vemb, vats, labels: List[str], device) -> str:
    completions = [(" " + s) if isinstance(s, str) and not s.startswith(" ") else str(s) for s in labels]
    scores = _score_completions_logprob_batched(model, tokenizer, prompt_ids, vemb, vats, completions, device, batch_size=min(32, len(completions)))
    best = int(torch.tensor(scores).argmax().item())
    return str(labels[best])


def _shortlist_labels_by_overlap(labels: List[str], question: str, topk: int = 50) -> List[str]:
    """
    Token-overlap shortlist for large label spaces.

    This does NOT change prompt length; it only reduces how many candidate labels we score.

    For very large label spaces (e.g., ordering), using a tiny topk can exclude the GT label,
    making accuracy artificially low. We enforce a safer minimum.
    """
    if topk <= 0 or len(labels) <= topk:
        return labels

    # safety: if label space is large, don't use a tiny topk
    if len(labels) >= 300 and topk < 150:
        topk = 150

    q = (question or "").lower()
    q_toks = set(re.findall(r"[a-z0-9_]+", q))

    scored = []
    for lab in labels:
        l = (lab or "").lower()
        l_toks = set(re.findall(r"[a-z0-9_]+", l))
        inter = len(q_toks & l_toks)

        bonus = 0
        if ("before" in q or "after" in q or "first" in q or "earlier" in q or "later" in q) and ("before" in l or "after" in l):
            bonus = 1

        scored.append((inter + bonus, lab))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [lab for _, lab in scored[:topk]]


def _extract_two_ordering_events(question: str, event_set: Optional[set]) -> Optional[Tuple[str, str]]:
    """
    Extract two event triplets from an ordering question.

    We assume event phrases are typically 3 tokens: '{instrument} {verb} {target}'
    (tokens may contain underscores). We validate using event_set derived from ordering vocab.

    Returns (e1, e2) if we can find two events; otherwise None.
    """
    if not question or not event_set:
        return None

    q = question.strip().lower()

    # Remove trailing instruction-like tail to reduce noise
    q = re.split(r"\b(answer|reply|respond)\b", q, maxsplit=1)[0]
    q = re.sub(r"\s+", " ", q).strip()

    toks = re.findall(r"[a-z0-9_]+", q)
    if len(toks) < 6:
        return None

    matches: List[Tuple[int, str]] = []

    # Prefer 3-token events (instrument verb target)
    for i in range(0, len(toks) - 2):
        phr = " ".join(toks[i:i+3])
        if phr in event_set:
            matches.append((i, phr))

    # Sometimes events could be 4 tokens (rare, but keep as fallback)
    if len(matches) < 2:
        for i in range(0, len(toks) - 3):
            phr = " ".join(toks[i:i+4])
            if phr in event_set:
                matches.append((i, phr))

    if len(matches) < 2:
        return None

    # Deduplicate by phrase, keep earliest position
    first_pos = {}
    for pos, phr in matches:
        if phr not in first_pos:
            first_pos[phr] = pos

    uniq = sorted([(pos, phr) for phr, pos in first_pos.items()], key=lambda x: x[0])
    if len(uniq) < 2:
        return None

    # Pick two phrases farthest apart (usually corresponds to the two compared events)
    best = None
    for a in range(len(uniq)):
        for b in range(a + 1, len(uniq)):
            dist = abs(uniq[b][0] - uniq[a][0])
            if best is None or dist > best[0]:
                best = (dist, uniq[a][1], uniq[b][1])

    if best is None:
        return None

    _, e1, e2 = best
    if not e1 or not e2 or e1 == e2:
        return None
    return e1, e2


def _structured_ordering_candidates(question: str, closed_labels: Dict[str, List[str]]) -> Optional[List[str]]:
    """
    Build structured candidates for ordering_text:
      - Parse two event phrases from the question
      - Return exactly 2 candidates: 'e1 before e2' and 'e2 before e1'

    If we cannot confidently parse 2 events, return None (caller will fall back).
    """
    ev_list = closed_labels.get("__ordering_events__", [])
    if not ev_list:
        return None

    event_set = set(ev_list)
    pair = _extract_two_ordering_events(question, event_set)
    if pair is None:
        return None

    e1, e2 = pair
    return [f"{e1} before {e2}", f"{e2} before {e1}"]


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
) -> List[str]:
    """
    decode_mode:
      - greedy: generate text
      - closed: closed-set decode for bool/count/choice and any category present in vocabs_json
      - hybrid: greedy; if cannot parse / not in closed set, fallback to closed
    """
    bsz = images.size(0)

    prompts = []
    for q, meta in zip(questions, metas):
        prompts.append(_build_prompt(q, meta, closed_labels=closed_labels))

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # model-specific: get video embeds/atts once
    with torch.no_grad():
        video_embeds, video_atts = model.encode_video(images.to(device))

    # --- greedy decode ---
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total_len = int(input_ids.size(1) + max_new_tokens)

    padded_ids = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    padded_att = torch.zeros((bsz, total_len), dtype=torch.long, device=device)

    padded_ids[:, : input_ids.size(1)] = input_ids
    padded_att[:, : input_ids.size(1)] = attn

    valid_lens = attn.sum(dim=1).long()
    finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    only_new = torch.empty((bsz, 0), dtype=torch.long, device=device)
    batch_idx = torch.arange(bsz, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            max_valid = int(valid_lens.max().item())
            if max_valid >= total_len:
                break

            ids_now = padded_ids[:, :max_valid]
            att_now = padded_att[:, :max_valid]

            logits = _forward_logits(model, ids_now, att_now, video_embeds, video_atts, video=None)
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
        greedy_texts.append(tokenizer.decode(toks, skip_special_tokens=True))

    outs: List[str] = []
    for i in range(bsz):
        q = questions[i]
        meta = metas[i]
        pred = greedy_texts[i].strip()

        lt = _infer_label_type_from_meta(q, meta)
        cat = (meta.get("category") or "").lower().strip()
        if cat == "phase":
            cat = "phase_transition"

        # Extract true prompt ids (no padding) for closed scoring
        prompt_att = attn[i]
        prompt_len = int(prompt_att.sum().item())
        prompt_ids_1 = input_ids[i, :prompt_len].contiguous()

        vemb = video_embeds[i:i+1]
        vats = video_atts[i:i+1]

        # ---- BOOL ----
        if lt == "bool":
            if decode_mode == "closed":
                outs.append(_closed_predict_bool(model, tokenizer, prompt_ids_1, vemb, vats, device))
            else:
                low = pred.lower()
                if any(x in low for x in ["yes", "true", "1"]):
                    outs.append("YES")
                elif any(x in low for x in ["no", "false", "0"]):
                    outs.append("NO")
                else:
                    outs.append(_closed_predict_bool(model, tokenizer, prompt_ids_1, vemb, vats, device))
            continue

        # ---- CHOICE ----
        if lt == "choice_index":
            K = meta.get("choice_K", None)
            if K is None:
                K = _parse_choice_K_from_question(q)

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
            # Special: ordering_text -> structured candidates (2-way classification)
            # If we can parse two events from the question, we only score:
            #   e1 before e2  vs  e2 before e1
            if cat == "ordering" and lt == "text":
                struct_cands = _structured_ordering_candidates(q, closed_labels)
                if struct_cands is not None and len(struct_cands) == 2:
                    labels_use = struct_cands
                    if decode_mode == "closed":
                        outs.append(_closed_predict_vocab_text(model, tokenizer, prompt_ids_1, vemb, vats, labels_use, device))
                    else:
                        low = pred.strip().lower()
                        if any(low == lab.strip().lower() for lab in labels_use):
                            outs.append(pred)
                        else:
                            outs.append(_closed_predict_vocab_text(model, tokenizer, prompt_ids_1, vemb, vats, labels_use, device))
                    continue

            # Default: vocab-based closed / approximate closed decoding
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
):
    references: List[Any] = []
    predictions: List[str] = []
    metas: List[Dict[str, Any]] = []

    closed_labels = _load_closed_labels(vocabs_json) if vocabs_json else None

    model.eval()
    for _, (images, questions, answers, batch_metas) in enumerate(tqdm(val_loader), 0):
        images = images.to(device)

        with torch.no_grad():
            gen = batch_decode(
                images=images,
                questions=[q[0] if isinstance(q, list) and q else q for q in questions],
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
            )
            references.extend(answers)
            predictions.extend(gen)
            metas.extend(batch_metas)

    print("First 5 GT:", references[:5])
    print("First 5 Pred:", predictions[:5])
    print("First 5 Meta:", metas[:5])
    return references, predictions, metas
