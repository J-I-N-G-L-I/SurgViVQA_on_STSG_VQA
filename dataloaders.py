import os
import json
import torch
from PIL import Image
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