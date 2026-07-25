import torch
from torch.utils.data import IterableDataset, DataLoader
from bit.tokenizer import BITTokenizer
from bit.config import BITConfig
import os

class BITDataset(IterableDataset):
    def __init__(self, file_path: str, tokenizer: BITTokenizer, config: BITConfig, split: str = "train"):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Determine file size to handle train/val split roughly
        self.file_size = os.path.getsize(file_path)
        self.train_size = int(0.9 * self.file_size)
        
    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            if self.split == "val":
                f.seek(self.train_size)
            
            buffer = ""
            while True:
                line = f.readline()
                if not line:
                    if self.split == "train":
                        # Loop back for training
                        f.seek(0)
                        continue
                    else:
                        break
                
                if self.split == "train" and f.tell() > self.train_size:
                    break
                    
                buffer += line
                if len(buffer) > self.config.block_size * 10: # Read in chunks
                    ids = self.tokenizer.encode(buffer, add_special_tokens=True)
                    for i in range(0, len(ids) - self.config.block_size - 1, self.config.block_size):
                        x = torch.tensor(ids[i:i+self.config.block_size], dtype=torch.long)
                        y = torch.tensor(ids[i+1:i+self.config.block_size+1], dtype=torch.long)
                        yield x, y
                    buffer = "" # Clear buffer after processing

def get_dataloader(file_path: str, tokenizer: BITTokenizer, config: BITConfig, split: str = "train"):
    dataset = BITDataset(file_path, tokenizer, config, split)
    return DataLoader(dataset, batch_size=config.batch_size, pin_memory=True)
