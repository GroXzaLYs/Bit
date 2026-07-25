import pytest
import torch
from bit.config import BITConfig
from bit.model import BitModel
from bit.tokenizer import BITTokenizer

def test_config():
    config = BITConfig()
    assert config.vocab_size == 10000
    assert config.n_layer == 24

def test_model_forward():
    config = BITConfig(n_layer=2, n_head=2, n_embed=32, vocab_size=100)
    model = BitModel(config)
    idx = torch.randint(0, 100, (1, 10))
    logits, loss, presents = model(idx)
    assert logits.shape == (1, 10, 100)
    assert loss is None
    assert presents is None

def test_model_with_cache():
    config = BITConfig(n_layer=2, n_head=2, n_embed=32, vocab_size=100)
    model = BitModel(config)
    idx = torch.randint(0, 100, (1, 10))
    logits, loss, presents = model(idx, use_cache=True)
    assert len(presents) == 2
    assert presents[0][0].shape == (1, 2, 10, 16) # (B, n_head, T, head_size)
