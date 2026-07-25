import os
import argparse
import torch
from bit.config import BITConfig
from bit.model import BitModel
from bit.tokenizer import BITTokenizer
from bit.trainer import BITTrainer
from bit.data import get_dataloader

def main():
    parser = argparse.ArgumentParser(description="BIT - Production Training")
    parser.add_argument("data", type=str, help="Path to training data (.txt)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size for BPE")
    args = parser.parse_args()

    config = BITConfig(vocab_size=args.vocab_size)
    
    # 1. Tokenizer
    tokenizer = BITTokenizer(config)
    if not os.path.exists(config.tokenizer_path):
        print("Training tokenizer...")
        tokenizer.train([args.data])
    
    # Update config vocab size based on trained tokenizer
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")

    # 2. Data
    train_loader = get_dataloader(args.data, tokenizer, config, split="train")
    val_loader = get_dataloader(args.data, tokenizer, config, split="val")

    # 3. Model
    model = BitModel(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params/1e6:.2f}M")

    # 4. Trainer
    trainer = BITTrainer(model, config)
    
    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_iter = trainer.load_checkpoint(args.resume)
        print(f"Starting from iteration {start_iter}")

    print("Starting training...")
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
