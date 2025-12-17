import os, json, argparse
from utils.stsg_metrics import evaluate_stsg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--vocabs_json", default=None)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    refs, preds, metas = [], [], []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            refs.append(obj["gt"])
            preds.append(obj["pred_raw"])
            metas.append({
                "video_id": obj.get("video_id"),
                "category": obj.get("category"),
                "answer_type": obj.get("answer_type"),
                "label_type": obj.get("label_type"),
                "answer_mode": obj.get("answer_mode"),
                "choice_K": obj.get("choice_K"),
                "choices": obj.get("choices", None),
            })

    report = evaluate_stsg(refs, preds, metas, vocabs_json=args.vocabs_json)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", args.out_json)

if __name__ == "__main__":
    main()
