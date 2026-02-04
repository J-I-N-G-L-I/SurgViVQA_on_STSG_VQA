# Fine-tuning checkpoints directory
# This folder stores model checkpoints during SSGVQA fine-tuning.
#
# Files saved:
#   - best_model.pth: Best model checkpoint (lowest validation loss)
#   - checkpoint_epochN.pth: Periodic checkpoints
#   - training_state.pth: Optimizer state and epoch info
#   - config.json: Training configuration
#   - train.log: Training log
#
# After training, evaluate with:
#   python fine-tune/eval_finetuned.py --checkpoint fine-tune/ckpt/best_model.pth
