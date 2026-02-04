"""
SSGVQA Dataset for SurgViVQA fine-tuning.

Loads QA pairs from SSGVQA txt files and prepares them for closed-set
supervised fine-tuning (SFT) compatible with the existing evaluation script.

Dataset format:
- QA root: /mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/
- Each video folder (VID02, VID22, ...) contains txt files (e.g., 102.txt)
- Each line: Question | Answer | other metadata...
- We use ONLY: Question (before 1st '|') and Answer (between 1st and 2nd '|')

Image mapping:
- Image root: /mnt/scratch/sc232jl/datasets/CholecT45/data
- <VID>/<frame_id padded to 6 digits>.png
- Example: VID02/102.txt -> VID02/000102.png
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class SSGVQASample:
    """Single QA sample from SSGVQA dataset."""
    video_id: str
    frame_id: str
    image_path: str
    question: str
    answer: str
    label_idx: int  # Index into 52-class label vocabulary


class SSGVQADataset(Dataset):
    """
    SSGVQA Dataset for fine-tuning SurgViVQA.
    
    This dataset loads QA pairs and prepares them for closed-set SFT training.
    The prompt format is designed to match the evaluation script exactly.
    
    Args:
        ssgvqa_root: Root directory of SSGVQA QA txt files
        image_root: Root directory of CholecT45 frame images
        video_ids: List of video IDs to include (e.g., ["VID02", "VID22"])
        processor: Image processor (e.g., from VideoMAE) for preprocessing
        label2idx: Label string -> index mapping (52-class)
        norm_label_map: Normalized label -> index mapping for fuzzy matching
        num_frames: Number of frames to use for video input (frames are repeated for static images)
        prompt_mode: "simple" or "choices" - controls prompt format
        max_samples: Optional limit on number of samples (for debugging)
        strict_missing_image: If True, raise error on missing images; if False, skip them
        skip_unknown_labels: If True, skip samples with answers not in label vocabulary
    """
    
    def __init__(
        self,
        ssgvqa_root: str,
        image_root: str,
        video_ids: List[str],
        processor,
        label2idx: Dict[str, int],
        norm_label_map: Dict[str, int],
        labels_list: List[str],
        num_frames: int = 16,
        prompt_mode: str = "simple",
        max_samples: Optional[int] = None,
        strict_missing_image: bool = False,
        skip_unknown_labels: bool = True,
    ):
        self.ssgvqa_root = ssgvqa_root
        self.image_root = image_root
        self.video_ids = video_ids
        self.processor = processor
        self.label2idx = label2idx
        self.norm_label_map = norm_label_map
        self.labels_list = labels_list
        self.num_frames = int(num_frames)
        self.prompt_mode = prompt_mode
        self.strict_missing_image = strict_missing_image
        self.skip_unknown_labels = skip_unknown_labels
        
        self._log_processor_once = True
        self.samples: List[SSGVQASample] = []
        self.skipped_unknown = 0
        self.skipped_missing_image = 0
        
        self._load_samples()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        logging.info(
            f"SSGVQADataset: loaded {len(self.samples)} samples from {len(video_ids)} videos. "
            f"Skipped: {self.skipped_unknown} unknown labels, {self.skipped_missing_image} missing images."
        )
    
    def _map_answer_to_idx(self, answer: str) -> int:
        """Map answer string to label index with fuzzy matching."""
        from labels import map_answer_to_label_idx
        return map_answer_to_label_idx(answer, self.label2idx, self.norm_label_map)
    
    def _load_samples(self):
        """Load all QA samples from txt files."""
        for vid in self.video_ids:
            vid_dir = os.path.join(self.ssgvqa_root, vid)
            if not os.path.isdir(vid_dir):
                logging.warning(f"Missing SSGVQA folder: {vid_dir}")
                continue
            
            # Get all txt files sorted by frame number
            txt_files = [f for f in os.listdir(vid_dir) if f.endswith(".txt")]
            txt_files = sorted(
                txt_files,
                key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
            )
            
            for fname in txt_files:
                frame_stem = os.path.splitext(fname)[0]
                
                # Convert frame number to 6-digit padded format for image path
                if frame_stem.isdigit():
                    frame_id = f"{int(frame_stem):06d}"
                else:
                    frame_id = frame_stem.zfill(6)
                
                img_path = os.path.join(self.image_root, vid, f"{frame_id}.png")
                
                # Check if image exists
                if not os.path.isfile(img_path):
                    if self.strict_missing_image:
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    else:
                        self.skipped_missing_image += 1
                        continue
                
                # Parse QA pairs from txt file
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
                        answer = parts[1]  # Use ONLY the answer between 1st and 2nd '|'
                        
                        if not question or not answer:
                            continue
                        
                        # Map answer to label index
                        label_idx = self._map_answer_to_idx(answer)
                        
                        if label_idx == -1:
                            if self.skip_unknown_labels:
                                self.skipped_unknown += 1
                                continue
                            else:
                                # Use -1 as unknown; will be filtered in training loop
                                pass
                        
                        self.samples.append(
                            SSGVQASample(
                                video_id=vid,
                                frame_id=frame_id,
                                image_path=img_path,
                                question=question,
                                answer=answer,
                                label_idx=label_idx,
                            )
                        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Process image through the VideoMAE processor."""
        if self.processor is None:
            raise RuntimeError("Processor is required for SurgViVQA fine-tuning.")
        
        try:
            processed = self.processor(images=image, return_tensors="pt")
        except TypeError:
            processed = self.processor(image, return_tensors="pt")
        
        pixel_values = processed.get("pixel_values", None) if isinstance(processed, dict) else getattr(processed, "pixel_values", None)
        if pixel_values is None:
            raise RuntimeError("Processor output missing 'pixel_values'.")
        
        if self._log_processor_once:
            self._log_processor_once = False
            logging.info(f"Processor: {self.processor.__class__.__name__} | pixel_values shape: {tuple(pixel_values.shape)}")
        
        return pixel_values.cpu().float()
    
    def _to_chw_frame(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert processor output to [3, H, W] format."""
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
    
    def build_prompt_prefix(self, question: str) -> str:
        """
        Build prompt prefix that matches the evaluation script format.
        
        The format MUST be compatible with:
        - utils/eval_surgvivqa_ssgvqa.py (build_prompt_prefix function)
        - Closed-set scoring via teacher-forcing log-prob
        """
        q = (question or "").strip()
        
        if self.prompt_mode == "choices":
            # Include all 52 labels as candidates (longer prompt)
            candidates = ", ".join(self.labels_list)
            return (
                f"Question: {q}\n"
                f"Candidates: {candidates}\n"
                "Answer:"
            )
        else:
            # Simple mode: short prefix only
            return (
                f"Question: {q}\n"
                "Answer:"
            )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, int, str, str]:
        """
        Returns:
            video_tensor: [T, 3, H, W] - repeated static frame for VideoMAE
            prompt_prefix: Prompt prefix (without the answer)
            target_text: Target text for training (" " + LABEL)
            label_idx: Label index (0 to 51)
            video_id: Video ID string
            frame_id: Frame ID string
        """
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample.image_path).convert("RGB")
        pixel_values = self._process_image(image)
        frame = self._to_chw_frame(pixel_values)  # [3, H, W]
        
        # Repeat single frame to T frames for VideoMAE input: [T, 3, H, W]
        video_tensor = frame.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        
        # Build prompt prefix (matches evaluation script format)
        prompt_prefix = self.build_prompt_prefix(sample.question)
        
        # Target text: " " + LABEL (leading space for GPT-2 tokenization compatibility)
        # This ensures tokenization matches evaluation script
        label_str = self.labels_list[sample.label_idx] if sample.label_idx >= 0 else sample.answer
        target_text = " " + label_str
        
        return video_tensor, prompt_prefix, target_text, sample.label_idx, sample.video_id, sample.frame_id


def collate_ssgvqa_train(batch):
    """
    Collate function for training dataloader.
    
    Returns:
        videos: [B, T, 3, H, W]
        prompt_prefixes: List[str] of length B
        target_texts: List[str] of length B (labels with leading space)
        label_indices: [B] tensor of label indices
        video_ids: List[str] of length B
        frame_ids: List[str] of length B
    """
    videos, prompts, targets, indices, vids, frames = [], [], [], [], [], []
    
    for v, p, t, i, vid, fid in batch:
        videos.append(v)
        prompts.append(p)
        targets.append(t)
        indices.append(i)
        vids.append(vid)
        frames.append(fid)
    
    videos = torch.stack(videos, dim=0)  # [B, T, 3, H, W]
    label_indices = torch.tensor(indices, dtype=torch.long)  # [B]
    
    return videos, prompts, targets, label_indices, vids, frames


def load_split_videos(split_file: str) -> List[str]:
    """Load video IDs from a JSON split file."""
    with open(split_file, "r", encoding="utf-8") as f:
        videos = json.load(f)
    return videos


def create_dataloaders(
    ssgvqa_root: str,
    image_root: str,
    splits_dir: str,
    processor,
    label2idx: Dict[str, int],
    norm_label_map: Dict[str, int],
    labels_list: List[str],
    num_frames: int = 16,
    prompt_mode: str = "simple",
    batch_size: int = 4,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders from split JSON files.
    
    Args:
        ssgvqa_root: Root directory of SSGVQA QA txt files
        image_root: Root directory of CholecT45 frame images
        splits_dir: Directory containing train_videos.json and val_videos.json
        processor: Image processor from VideoMAE
        label2idx: Label -> index mapping
        norm_label_map: Normalized label -> index mapping
        labels_list: List of 52 label strings
        num_frames: Number of frames for video input
        prompt_mode: "simple" or "choices"
        batch_size: Batch size
        num_workers: Number of dataloader workers
        max_train_samples: Optional limit for training samples (debugging)
        max_val_samples: Optional limit for validation samples (debugging)
    
    Returns:
        train_loader, val_loader
    """
    train_videos = load_split_videos(os.path.join(splits_dir, "train_videos.json"))
    val_videos = load_split_videos(os.path.join(splits_dir, "val_videos.json"))
    
    train_dataset = SSGVQADataset(
        ssgvqa_root=ssgvqa_root,
        image_root=image_root,
        video_ids=train_videos,
        processor=processor,
        label2idx=label2idx,
        norm_label_map=norm_label_map,
        labels_list=labels_list,
        num_frames=num_frames,
        prompt_mode=prompt_mode,
        max_samples=max_train_samples,
    )
    
    val_dataset = SSGVQADataset(
        ssgvqa_root=ssgvqa_root,
        image_root=image_root,
        video_ids=val_videos,
        processor=processor,
        label2idx=label2idx,
        norm_label_map=norm_label_map,
        labels_list=labels_list,
        num_frames=num_frames,
        prompt_mode=prompt_mode,
        max_samples=max_val_samples,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ssgvqa_train,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ssgvqa_train,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of dataset loading
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Import labels
    from labels import SSGVQA_LABELS, LABEL2IDX, NORM_LABEL_MAP
    
    # Test paths (adjust as needed)
    ssgvqa_root = "/mnt/scratch/sc232jl/datasets/SSGVQA/ssg-qa/"
    image_root = "/mnt/scratch/sc232jl/datasets/CholecT45/data"
    
    # Test with one video
    test_videos = ["VID02"]
    
    print(f"Testing dataset with videos: {test_videos}")
    print(f"SSGVQA root: {ssgvqa_root}")
    print(f"Image root: {image_root}")
    
    # Note: processor would be loaded from model in actual usage
    print("Skipping actual dataset test (requires processor).")
