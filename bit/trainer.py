import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import math
import time
from bit.config import BITConfig
from bit.model import BitModel

class BITTrainer:
    def __init__(self, model: BitModel, config: BITConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def save_checkpoint(self, it, val_loss, is_best=False):
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "iter": it,
            "config": self.config,
            "val_loss": val_loss
        }
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{it}.pt")
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, self.config.model_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["iter"]

    def train(self, train_loader, val_loader):
        self.model.train()
        iter_num = 0
        
        train_iter = iter(train_loader)
        
        while iter_num < self.config.max_iters:
            # Update learning rate
            lr = self.get_lr(iter_num)
            for param_group in self.optimizer.param_group:
                param_group["lr"] = lr
            
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
                
            x, y = x.to(self.config.device), y.to(self.config.device)
            
            t0 = time.time()
            logits, loss, _ = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            t1 = time.time()
            
            if iter_num % 10 == 0:
                self.writer.add_scalar("loss/train", loss.item(), iter_num)
                self.writer.add_scalar("lr", lr, iter_num)
                print(f"iter {iter_num}: loss {loss.item():.4f}, time {1000*(t1-t0):.2f}ms")

            if iter_num % self.config.eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                self.writer.add_scalar("loss/val", val_loss, iter_num)
                perplexity = math.exp(val_loss)
                self.writer.add_scalar("perplexity/val", perplexity, iter_num)
                print(f"step {iter_num}: val loss {val_loss:.4f}, perplexity {perplexity:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(iter_num, val_loss, is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= 5:
                    print("Early stopping triggered")
                    break
                    
            iter_num += 1
        
        self.writer.close()

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        losses = []
        val_iter = iter(val_loader)
        for _ in range(self.config.eval_iters):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            x, y = x.to(self.config.device), y.to(self.config.device)
            _, loss, _ = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses) if losses else 0
