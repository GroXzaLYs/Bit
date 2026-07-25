from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BITConfig:
    # Model architecture
    vocab_size: int = 10000
    block_size: int = 2048
    n_embed: int = 2048
    n_head: int = 16
    n_layer: int = 24
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_iters: int = 100000
    eval_interval: int = 1000
    eval_iters: int = 200
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 1e-5
    
    # System
    device: str = "cuda"
    seed: int = 1337
    
    # Paths
    model_path: str = "bit_model.pt"
    tokenizer_path: str = "tokenizer.json"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    # Inference
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8
