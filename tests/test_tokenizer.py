import pytest
import os
from bit.config import BITConfig
from bit.tokenizer import BITTokenizer

def test_tokenizer_train_and_encode(tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("Hello world! This is a test for the BIT tokenizer refactor. " * 10)
    
    config = BITConfig(tokenizer_path=str(tmp_path / "tokenizer.json"), vocab_size=100)
    tokenizer = BITTokenizer(config)
    
    tokenizer.train([str(data_file)])
    assert os.path.exists(config.tokenizer_path)
    
    encoded = tokenizer.encode("Hello world")
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    
    decoded = tokenizer.decode(encoded)
    assert "hello world" in decoded.lower()
