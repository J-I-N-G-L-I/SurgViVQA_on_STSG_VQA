import os
import re
import json
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class REALColonVideoQA(Dataset):
    def __init__(self, folder_name, sequences, type="in_template", transform=None, processor=None):

        self.sequences = sequences
        self.folder_name = folder_name
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.processor = processor

        with open(os.path.join(self.folder_name, "annotations", f"{type}.jsonl"), 'r') as f:
          lines = f.readlines()

        self.vqas = []
        for line in lines:
            entry = json.loads(line)
            if entry.get("video_id") in self.sequences:
                self.vqas.append([entry.get("video_id"), entry.get("frames"), entry.get("question"),
                                entry.get("answer"), entry.get("short_answer")])

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):

        video_folder = os.path.join(self.folder_name, "videos", self.vqas[idx][0] + '_frames')
        frame_names = self.vqas[idx][1]

        # Load frames
        frames = [Image.open(os.path.join(video_folder, frame_name +'.jpg')).convert('RGB') for frame_name in frame_names]
        # Process frames
        if self.processor:
            processed = self.processor(videos=[frames], return_tensors="pt")
            video_tensor = processed["pixel_values"].squeeze(0)
        else:
            # Apply transforms and stack into video tensor
            transformed_frames = [self.transform(img) for img in frames]
            video_tensor = torch.stack(transformed_frames)  # shape: [sequence_length, C, H, W]

        # question, answer and keyword
        question, answer, keyword = [self.vqas[idx][2]], [self.vqas[idx][3]], self.vqas[idx][4]

        # print(keyword)
        keyword = [[kw.strip() for kw in keyword.split(',')]]

        # return video, question, answer, keyword
        return video_tensor, question, answer, keyword


class EndoVis18VideoQA(Dataset):
    def __init__(self, sequences, folder_name, folder_tail,
                 transform=None, processor=None, sequence_length=8):
        self.sequences = sequences
        self.folder_name = folder_name
        self.folder_tail = folder_tail
        self.sequence_length = sequence_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.processor = processor
        self.samples = self._build_samples()
        
    def _build_samples(self):
        samples = []
        for seq in self.sequences:
            # Construct sequence folder path
            seq_folder = f'seq_{seq}'
            image_folder = os.path.join(self.folder_name, seq_folder, 'left_fr')
            qa_folder = os.path.join(self.folder_name, seq_folder, self.folder_tail)
            keyword_folder = os.path.join(self.folder_name, seq_folder, 'vqa/Keyword')

            # Get sorted frame files
            frame_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
            frame_ids = sorted([f.split('.')[0] for f in frame_files])

            # Create sequences of frames
            for i in range(len(frame_ids) - self.sequence_length + 1):
                chunk = frame_ids[i:i+self.sequence_length]
                
                # Build frame paths
                frame_paths = [os.path.join(image_folder, f"{fid}.png") for fid in chunk]
                
                # QA file corresponds to the last frame in the sequence
                qa_path = os.path.join(qa_folder, f"{chunk[-1]}_QA.txt")

                # Keyord file corresponds to the last frame in the sequence
                keyword_path = os.path.join(keyword_folder, f"{chunk[-1]}_QA.txt")

                if os.path.isfile(qa_path) and os.path.isfile(qa_path):
                    samples.append((frame_paths, qa_path, keyword_path))
                    
        return samples

    def _load_qa(self, qa_path):
        qa_pairs = []
        
        with open(qa_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and "|" in line:
                    question, answer = line.split("|", 1)
                    qa_pairs.append((question.strip(), answer.strip()))
        
        return qa_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, qa_path, keyword_path = self.samples[idx]
        
        # Load frames
        frames = [Image.open(p).convert('RGB') for p in frame_paths]

        # Process frames
        if self.processor:
            processed = self.processor(videos=[frames], return_tensors="pt")
            video_tensor = processed["pixel_values"].squeeze(0)
        else:
            # Apply transforms and stack into video tensor
            transformed_frames = [self.transform(img) for img in frames]
            video_tensor = torch.stack(transformed_frames)  # shape: [sequence_length, C, H, W]
        
        # Load QA pairs
        qa_pairs = self._load_qa(qa_path)
        questions = [q for q, a in qa_pairs]
        answers = [a for q, a in qa_pairs]
        keywords = [[kw.strip() for kw in k.split(' , ')] for _, k in self._load_qa(keyword_path)]
        
        return video_tensor, questions, answers, keywords


def collate_qa_clipwise(batch):
    """
    batch: list of (imgs, questions, answers)
    returns: (videos, flat_questions, flat_answers, flat_keywords)
    """
    vids, flat_q, flat_a, flat_k = [], [], [], []
    for video, qs, as_, ks in batch:
        for q,a,k in zip(qs, as_, ks):
            vids.append(video)
            flat_q.append(q)
            flat_a.append(a)
            flat_k.append(k)
    videos = torch.stack(vids, dim=0)           # (N, T, 3, 224, 224)
    return videos, flat_q, flat_a, flat_k


class STSGTemporalVideoQA(Dataset):
    """
    Load STSG temporal QA from:
      stsg_qa_root/VIDxx/temporal_qa.json
    and frames from:
      frame_root/VIDxx/000703.png

    Each sample uses evidence.keyframes (or scope.frames fallback),
    then resamples/pads to num_frames (default=8) for VideoMAE.
    """
    def __init__(
        self,
        video_ids: List[str],
        stsg_qa_root: str,
        frame_root: str,
        num_frames: int = 8,
        transform=None,
        processor=None,
        max_samples: Optional[int] = None,
        strict_missing_frame: bool = False,
        include_rationale: bool = False,
        rationale_key: str = "auto",
        force_mode: str = "",
    ):
        self.video_ids = video_ids
        self.stsg_qa_root = stsg_qa_root
        self.frame_root = frame_root
        self.num_frames = num_frames
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.processor = processor
        self.strict_missing_frame = strict_missing_frame

        self.include_rationale = include_rationale
        self.rationale_key = rationale_key
        self.force_mode = force_mode

        self.samples: List[Dict[str, Any]] = []
        for vid in self.video_ids:
            qa_path = os.path.join(self.stsg_qa_root, vid, "temporal_qa.json")
            if not os.path.isfile(qa_path):
                raise FileNotFoundError(f"Missing temporal_qa.json: {qa_path}")
            with open(qa_path, "r") as f:
                data = json.load(f)

            # temporal_qa.json is usually a list; but be tolerant
            if isinstance(data, dict) and "qas" in data:
                qas = data["qas"]
            elif isinstance(data, list):
                qas = data
            else:
                raise ValueError(f"Unsupported temporal_qa.json format in {qa_path}")

            for qa in qas:
                self.samples.append({
                    "video_id": vid,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", None),
                    "answer_type": qa.get("answer_type", None),
                    "category": qa.get("category", qa.get("evidence", {}).get("category", None)),
                    "evidence": qa.get("evidence", {}),
                    "raw": qa,
                })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resample_to_k(frames: List[int], k: int) -> List[int]:
        if len(frames) == 0:
            return [0] * k
        if len(frames) == k:
            return frames
        # linspace index on the list itself (allows duplicates if len(frames)<k)
        idx = np.linspace(0, len(frames) - 1, k)
        idx = np.round(idx).astype(int).tolist()
        return [frames[i] for i in idx]

    def _get_keyframes(self, sample: Dict[str, Any]) -> List[int]:
        ev = sample.get("evidence", {}) or {}
        kfs = ev.get("keyframes", None)
        if isinstance(kfs, list) and len(kfs) > 0:
            return [int(x) for x in kfs]

        # fallback: scope.frames = [start, end]
        scope = ev.get("scope", {}) or {}
        fr = scope.get("frames", None)
        if isinstance(fr, list) and len(fr) == 2:
            s, e = int(fr[0]), int(fr[1])
            if e < s:
                s, e = e, s
            # take up to k frames uniformly from [s,e]
            count = min(self.num_frames, max(1, e - s + 1))
            xs = np.linspace(s, e, count)
            xs = np.round(xs).astype(int).tolist()
            return xs

        return [0]

    def _frame_path(self, vid: str, frame_id: int) -> str:
        # default: 6-digit png
        p = os.path.join(self.frame_root, vid, f"{frame_id:06d}.png")
        if os.path.isfile(p):
            return p
        # tolerate jpg (some exports differ)
        p2 = os.path.join(self.frame_root, vid, f"{frame_id:06d}.jpg")
        if os.path.isfile(p2):
            return p2
        return p  # return default path (for error message)

    @staticmethod
    def _pick_rationale(raw: dict, category: str, rationale_key: str):
        if not raw:
            return None
        if rationale_key and rationale_key != "auto":
            return raw.get(rationale_key)

        # auto：优先按类别找 e.g. rationale_duration_v1_en
        c = (category or "").lower()
        cand = [
            f"rationale_{c}_v1_en",
            f"rationale_{c}_en",
        ]
        for k in cand:
            if k in raw:
                return raw[k]

        # 再退一步：找任意 rationale_*_en
        for k, v in raw.items():
            if isinstance(k, str) and k.startswith("rationale_") and k.endswith("_en"):
                return v
        return None


    @staticmethod
    def _parse_choice_K_from_question(q: str) -> Optional[int]:
        """
        从 question 尾部模板提取 K：
        - Answer with a number 1-5
        - Answer 1-3
        - Reply 1-4
        """
        if not q:
            return None
        qq = q.strip().lower()

        # 常见三种模板都覆盖；允许 1 - 5 / 1-5 / 1 to 5
        m = re.search(r"(answer|reply)\s*(with\s*(a\s*)?number\s*)?1\s*(\-|–|to)\s*(\d+)\b", qq)
        if m:
            try:
                return int(m.group(5))
            except Exception:
                return None
        return None

    @staticmethod
    def _looks_like_bool_question(q: str) -> bool:
        if not q:
            return False
        qq = q.lower()
        # 你生成模板里通常会出现这些提示词（可继续加）
        hints = [
            "yes or no",
            "true or false",
            "answer yes",
            "answer no",
            "answer with yes",
            "answer with no",
            "answer with true",
            "answer with false",
            "reply yes",
            "reply no",
            "reply true",
            "reply false",
        ]
        return any(h in qq for h in hints)

    @staticmethod
    def _normalize_answer_type(at: Any) -> str:
        if at is None:
            return ""
        s = str(at).strip().lower()
        # 兼容你数据里可能出现的写法
        if s in ("bool", "boolean", "binary", "bin", "yesno", "yes/no"):
            return "bool"
        if s in ("choice", "mcq", "multi_choice", "multichoice", "option", "index"):
            return "choice_index"
        if s in ("count",):
            return "count"
        if s in ("duration",):
            return "duration"
        if s in ("seconds", "sec", "time", "reg", "numeric", "number", "float", "int"):
            return "seconds"
        if s in ("cat", "text", "string", "phrase"):
            return "text"
        return s

    def _infer_label_type_and_k(self, raw: dict, category: str) -> Tuple[str, Optional[int]]:
        """
        返回 (label_type, choice_K)
        label_type ∈ {bool, count, duration, seconds, choice_index, text}
        """
        q = (raw.get("question") or "")
        ans = raw.get("answer", None)
        at_norm = self._normalize_answer_type(raw.get("answer_type"))

        cat = (category or raw.get("category") or raw.get("evidence", {}).get("category") or "").lower()
        if cat == "phase":
            cat = "phase_transition"

        # 允许外部强制（用于 debug）
        if self.force_mode:
            forced = self._normalize_answer_type(self.force_mode)
            if forced == "choice_index":
                return "choice_index", self._parse_choice_K_from_question(q)
            return forced, None

        # 1) count/duration 最先兜住（避免 0/1 被当 bool）
        if cat == "count":
            return "count", None
        if cat == "duration":
            return "duration", None

        # 2) 先用 question 模板判定 choice（最可靠）
        K = self._parse_choice_K_from_question(q)
        if isinstance(K, int) and K > 0:
            return "choice_index", K

        # 3) answer_type 若明确，直接用
        if at_norm in ("bool", "choice_index", "seconds", "text"):
            if at_norm == "choice_index":
                return "choice_index", raw.get("choice_K", None) or K
            return at_norm, None

        # 4) motion 永远是 text（你的定义是 9 类字符串）
        if cat == "motion":
            return "text", None

        # 5) bool：question 提示词 或 答案本身 bool
        if self._looks_like_bool_question(q) or isinstance(ans, bool):
            return "bool", None

        # 6) 如果答案是 0/1 且属于“可能有 bin 子类”的类别，优先当 bool（除非它是 choice 且 question 已匹配 1-K）
        if isinstance(ans, (int, np.integer)) and int(ans) in (0, 1) and cat in (
            "ordering", "boundary", "concurrency", "extreme", "phase_transition"
        ):
            return "bool", None

        # 7) 剩下 numeric → seconds；str → text
        if isinstance(ans, (int, float, np.integer, np.floating)):
            return "seconds", None
        return "text", None

    def __getitem__(self, idx):
        s = self.samples[idx]
        vid = s["video_id"]

        keyframes = self._get_keyframes(s)
        sel = self._resample_to_k(keyframes, self.num_frames)

        frames: List[Image.Image] = []
        for fid in sel:
            fp = self._frame_path(vid, int(fid))
            if not os.path.isfile(fp):
                if self.strict_missing_frame:
                    raise FileNotFoundError(f"Missing frame: {fp}")
                # fallback: use a black image
                frames.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                continue
            frames.append(Image.open(fp).convert("RGB"))

        if self.processor:
            # VideoMAE's AutoImageProcessor expects "images", not "videos"
            processed = self.processor(images=frames, return_tensors="pt")
            pv = processed["pixel_values"]  # could be (T,3,H,W) or (1,T,3,H,W)
            if pv.dim() == 5:
                video_tensor = pv.squeeze(0)
            elif pv.dim() == 4:
                video_tensor = pv
            else:
                raise RuntimeError(f"Unexpected pixel_values shape: {pv.shape}")
        else:
            transformed = [self.transform(im) for im in frames]
            video_tensor = torch.stack(transformed, dim=0)

        raw = s["raw"]
        category = (s.get("category") or "").lower()
        if category == "phase":
            category = "phase_transition"

        label_type, choice_K = self._infer_label_type_and_k(raw, category)

        rationale = self._pick_rationale(raw, category, self.rationale_key) if self.include_rationale else None

        meta = {
            "video_id": vid,
            "keyframes": sel,
            "category": category,
            "answer_type": s.get("answer_type"),
            # 新字段：推理/评测统一读它
            "label_type": label_type,
            "choice_K": choice_K,
            # 兼容旧字段（你之前代码里叫 answer_mode）
            "answer_mode": label_type,
            "rationale": rationale,
            "choices": raw.get("choices") or raw.get("options") or raw.get("choice_candidates") or None,
            "qa_raw": raw,
        }
        return video_tensor, s["question"], s["answer"], meta


def collate_stsg(batch):
    videos, questions, answers, metas = [], [], [], []
    for v, q, a, m in batch:
        videos.append(v)
        questions.append(q)
        answers.append(a)
        metas.append(m)
    videos = torch.stack(videos, dim=0)  # (B,T,3,H,W)
    return videos, questions, answers, metas

