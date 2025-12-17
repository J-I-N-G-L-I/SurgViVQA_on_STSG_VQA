import torch
import torch.nn.functional as F

@torch.no_grad()
def nll_for_candidate(model, tokenizer, video, prompt_text: str, cand_text: str, device):
    """
    NLL of cand_text given prompt_text under SurgViVQA (video conditioned).
    """
    full = prompt_text + cand_text
    enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # 计算 answer 起始位置
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ans_start = prompt_ids.size(1)

    # forward: logits shape [1, L, vocab]
    logits = model(video=video, qa_inputs_ids=input_ids, qa_att_mask=attn)

    # 语言模型预测 token t 用 logits[t-1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # 只算答案部分的 loss（从 ans_start 开始的 token）
    # 注意 shift 后对齐：labels 的 index i 对应原始 token i+1
    start = max(ans_start - 1, 0)
    cand_logits = shift_logits[:, start:, :]
    cand_labels = shift_labels[:, start:]

    loss = F.cross_entropy(
        cand_logits.reshape(-1, cand_logits.size(-1)),
        cand_labels.reshape(-1),
        reduction="sum"
    )
    return float(loss.item())

@torch.no_grad()
def predict_closed_set(model, tokenizer, video, question: str, candidates: list, device):
    prompt = f"Question: {question}\nAnswer:"
    best, best_loss = None, 1e30
    for c in candidates:
        loss = nll_for_candidate(model, tokenizer, video, prompt, " " + c, device)
        if loss < best_loss:
            best_loss, best = loss, c
    return best, best_loss
