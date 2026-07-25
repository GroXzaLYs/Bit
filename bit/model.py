import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from bit.config import BITConfig

class SwiGLU(nn.Module):
    def forward(self, x):
        return F.silu(x[..., :x.shape[-1]//2]) * x[..., x.shape[-1]//2:]

class FeedForward(nn.Module):
    def __init__(self, config: BITConfig):
        super().__init__()
        hidden_dim = int(4 * config.n_embed * 2 / 3)
        self.w1 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w3(self.act(torch.cat([self.w1(x), self.w2(x)], dim=-1))))

class Block(nn.Module):
    def __init__(self, config: BITConfig):
        super().__init__()
        self.config = config
        head_size = config.n_embed // config.n_head
        # Using a vectorized attention for better KV-cache support
        self.n_head = config.n_head
        self.head_size = head_size
        
        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.k_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.v_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.o_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None, use_cache=False):
        B, T, C = x.shape
        
        # Attention
        norm_x = self.ln1(x)
        q = self.q_proj(norm_x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        present = (k, v) if use_cache else None
        
        T_total = k.size(-2)
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.tril[:, :, T_total-T:T_total, :T_total] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        
        x = x + y
        
        # FeedForward
        x = x + self.ffwd(self.ln2(x))
        
        return x, present

class BitModel(nn.Module):
    def __init__(self, config: BITConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        device = idx.device
        B, T = idx.shape
        
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
        
        pos = torch.arange(past_length, past_length + T, dtype=torch.long, device=device).unsqueeze(0)
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, presents
