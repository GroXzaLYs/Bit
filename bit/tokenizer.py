import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from bit.config import BITConfig

class BITTokenizer:
    def __init__(self, config: BITConfig):
        self.config = config
        if os.path.exists(config.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(config.tokenizer_path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            self.tokenizer.pre_tokenizer = Whitespace()
            
    def train(self, files: list[str]):
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
            show_progress=True
        )
        self.tokenizer.train(files, trainer)
        self.tokenizer.save(self.config.tokenizer_path)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        if add_special_tokens:
            # Simple BOS/EOS addition if not already there
            bos_id = self.tokenizer.token_to_id("<BOS>")
            eos_id = self.tokenizer.token_to_id("<EOS>")
            return [bos_id] + ids + [eos_id]
        return ids
    
    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def get_pad_id(self) -> int:
        return self.tokenizer.token_to_id("<PAD>")

    def get_eos_id(self) -> int:
        return self.tokenizer.token_to_id("<EOS>")
    
    def get_bos_id(self) -> int:
        return self.tokenizer.token_to_id("<BOS>")
