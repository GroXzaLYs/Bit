import torch
from torch.nn import functional as F
from bit.model import BitModel
from bit.config import BITConfig
from bit.tokenizer import BITTokenizer

class BITInference:
    def __init__(self, model: BitModel, tokenizer: BITTokenizer, config: BITConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        device = next(self.model.parameters()).device
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        
        past_key_values = None
        generated_ids = []
        
        # Prefill
        logits, _, past_key_values = self.model(idx, use_cache=True)
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        generated_ids.append(idx_next.item())
        
        # Generation with KV-cache
        for _ in range(max_new_tokens - 1):
            if idx_next.item() == self.tokenizer.get_eos_id():
                break
                
            logits, _, past_key_values = self.model(idx_next, past_key_values=past_key_values, use_cache=True)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated_ids.append(idx_next.item())
            
        return self.tokenizer.decode(generated_ids)

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], max_new_tokens: int = 100, temperature: float = 1.0):
        # Simplified batch inference
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, max_new_tokens, temperature))
        return results
