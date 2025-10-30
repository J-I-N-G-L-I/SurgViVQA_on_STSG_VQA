import os
import torch
import random
import argparse
import numpy as np

from dataloaders import REALColonVideoQA, EndoVis18VideoQA, collate_qa_clipwise
from torch.utils.data import DataLoader, Subset

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.model import SurgViVQA
from peft import TaskType, LoraConfig

from utils.inference import inference
from utils.metrics import evaluate_model

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
    parser.add_argument('--dataset',        default='realcolon',  help='Dataset name: realcolon / endovis')
    parser.add_argument('--checkpoint_dir', default='SurgViVQA_checkpoint/',  help='path to checkpoint')

    # Inference parameters
    parser.add_argument('--batch_size',     type=int,   default=16,               help='batch size')
    parser.add_argument('--workers',        type=int,   default=8,                help='for data-loading')
    parser.add_argument('--random_seed',    type=int,   default=42,               help='random seed')
    parser.add_argument('--seq_length',     type=int,   default=64,               help='sequence length for question and answer')

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

    # Inference parameters
    print(f'Batch size: {args.batch_size}')
    print(f'Sequence length: {args.seq_length}')

    args.checkpoint_dir = f'{args.checkpoint_dir}/{args.dataset}'
    os.makedirs(args.checkpoint_dir, exist_ok = True)
      
    # GPT2 used as LLM
    pretrained_model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    decoder_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
    )

    model = SurgViVQA(device=device, tokenizer=tokenizer, decoder_model=decoder_model, peft_config=lora_config)
    # model.load_state_dict(torch.load(f'{args.checkpoint_dir}/best_model.pth', map_location=device))
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Model params: ', pytorch_total_params)

    # ----------------------
    # INFERENCE: IN-TEMPLATE
    # ----------------------

    if args.dataset == 'realcolon':

        # Path to the REAL-Colon-VQA dataset
        folder_name = '/SAN/medic/Kvasir/REAL-Colon-VQA' # to substitute with actual dataset location

        val_seq = ['002-006']

        val_dataset = REALColonVideoQA(
            folder_name = f'{folder_name}/dataset',
            sequences = val_seq,
            type = "in_template"
            )
    elif args.dataset == 'endovis':

        # Path to the EndoVis-18-VQA dataset
        folder_name = '/SAN/medic/Kvasir/EndoVis-18-VQA' # to substitute with actual dataset location
        folder_tail = '/vqa/Sentence'

        val_seq = [1, 5, 16]

        val_dataset = EndoVis18VideoQA(
            sequences = val_seq,
            folder_name = f'{folder_name}/dataset',
            folder_tail = folder_tail
            )

    # Dataloader
    print("=" * 50 + "\n" + "IN-TEMPLATE EVALUATION".center(50) + "\n" + "=" * 50)

    val_dataloader = DataLoader(dataset=val_dataset if not args.debug else Subset(val_dataset, list(range(10))), 
                                batch_size=args.batch_size, collate_fn=collate_qa_clipwise,
                                shuffle=False, num_workers=args.workers)
    print('Val sample size: ', len(val_dataset))

    references, predictions, keyword_references = inference(val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device, seq_length=args.seq_length)
    
    # Evaluate model predictions against ground truth for in-template setting
    evaluate_model(references, predictions, keyword_references)

    # -----------------------
    # INFERENCE: OUT-TEMPLATE
    # -----------------------

    if args.dataset == 'realcolon':

        val_dataset = REALColonVideoQA(
            folder_name = f'{folder_name}/dataset',
            sequences = val_seq,
            type = "out_template"
           )
    elif args.dataset == 'endovis':

        val_dataset = EndoVis18VideoQA(
            sequences = val_seq,
            folder_name = f'{folder_name}/dataset_new',
            folder_tail = folder_tail
            )
   
    # Dataloader
    print("=" * 50 + "\n" + "OUT-TEMPLATE EVALUATION".center(50) + "\n" + "=" * 50)
    
    val_dataloader = DataLoader(dataset=val_dataset if not args.debug else Subset(val_dataset, list(range(10))), batch_size=args.batch_size, collate_fn=collate_qa_clipwise,
                                shuffle=False, num_workers=args.workers)
    print('Val sample size: ', len(val_dataset))

    references, predictions, keyword_references = inference(val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device, seq_length=args.seq_length)

    # Evaluate model predictions against ground truth for out-template setting
    evaluate_model(references, predictions, keyword_references)
