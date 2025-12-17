import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from dataloaders import REALColonVideoQA, EndoVis18VideoQA, collate_qa_clipwise
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.model import SurgViVQA
from peft import TaskType, LoraConfig

from utils.trainer import train_val

import warnings
warnings.filterwarnings('ignore')


# point all caches into your models folder
base_cache = "hf_models"
os.environ["HF_HOME"]            = f"{base_cache}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{base_cache}/transformers"
os.environ["TORCH_HOME"]         = f"{base_cache}/torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description='VideoQuestionAnswerGeneration')

    # Dataset and checkpoint configuration
    parser.add_argument('--dataset',        default='realcolon', choices=['realcolon', 'endovis'], help='Dataset name: realcolon / endovis')
    parser.add_argument('--checkpoint_dir', default='SurgViVQA_checkpoint/',  help='path to checkpoint')

    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=60,               help='number of epochs to train for')
    parser.add_argument('--batch_size',     type=int,   default=16,               help='batch size')
    parser.add_argument('--lr',             type=float, default=0.0000002,        help='0.0000001, 0.00000005')
    parser.add_argument('--seq_length',     type=int,   default=64,               help='sequence length for question and answer')

    parser.add_argument('--random_seed',    type=int,   default=42,               help='random seed')
    parser.add_argument('--workers',        type=int,   default=8,                help='for data-loading')

    # For debugging
    parser.add_argument('--debug',          action='store_true',                  help='Enable debug mode to use fewer samples')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    
    # Set random seed for reproducibility
    seed_everything(args.random_seed)
    print(f'Random seed: {args.random_seed}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    # Dataset used
    print(f'Dataset used: {args.dataset}')

    # Training parameters
    print(f'Number of epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Sequence length: {args.seq_length}')

    args.checkpoint_dir = f'{args.checkpoint_dir}/{args.dataset}'
    os.makedirs(args.checkpoint_dir, exist_ok = True)
    
    if args.dataset == 'realcolon':

        # Path to the REAL-Colon-VQA dataset
        # folder_name = '/SAN/medic/Kvasir/REAL-Colon-VQA' # to substitute with actual dataset location

        folder_name = '/mnt/scratch/sc232jl/SurgViVQA/REAL-Colon-VQA_annotations'

        train_seq = ['002-001', '002-002', '002-003', '002-004', '002-005']
        val_seq = ['002-006']

        train_dataset = REALColonVideoQA(
            folder_name = f'{folder_name}/dataset',
            sequences = train_seq,
            type = "in_template",
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor()
            ])) 

        val_dataset = REALColonVideoQA(
            folder_name = f'{folder_name}/dataset',
            sequences = val_seq,
            type = "in_template",
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]))
    elif args.dataset == 'endovis':

        # Path to the EndoVis-18-VQA dataset
        folder_name = '/SAN/medic/Kvasir/EndoVis-18-VQA' # to substitute with actual dataset location
        folder_tail = 'vqa/Sentence'

        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]

        train_dataset = EndoVis18VideoQA(
            sequences = train_seq,
            folder_name = f'{folder_name}/dataset',
            folder_tail = folder_tail,
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor()
            ]))    
        val_dataset = EndoVis18VideoQA(
            sequences = val_seq,
            folder_name = f'{folder_name}/dataset',
            folder_tail = folder_tail,
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor()
            ]))

    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset if not args.debug else Subset(train_dataset, list(range(10))),
                                batch_size=args.batch_size, collate_fn=collate_qa_clipwise,
                                shuffle=True, num_workers=args.workers)
    val_dataloader = DataLoader(dataset=val_dataset if not args.debug else Subset(val_dataset, list(range(10))), 
                                batch_size=args.batch_size, collate_fn=collate_qa_clipwise,
                                shuffle=False, num_workers=args.workers)
    
    print('Train sample size: ', len(train_dataset), 'Val sample size: ', len(val_dataset))
    
    # GPT2 used as LLM
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    decoder_model = AutoModelForCausalLM.from_pretrained('gpt2')

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )

    model = SurgViVQA(device=device, tokenizer=tokenizer, decoder_model=decoder_model, peft_config=lora_config)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Model params: ', pytorch_total_params)

    # init optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    # train and validation
    train_val(device=device, args=args, model=model, tokenizer=tokenizer, optimizer=optimizer, criterion=criterion,
               train_dataloader=train_dataloader, val_dataloader=val_dataloader, debug_flag=args.debug)