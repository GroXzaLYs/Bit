#!/usr/bin/env python3
"""
BIT - Chat Interface
----------------------
Memuat model BIT yang sudah dilatih (bit_model.pt) dan mengobrol dengannya.
100% lokal, tanpa API key, tanpa internet, tanpa model pihak lain.

Cara pakai:
  python bit_chat.py

Perintah:
  /temp <angka>   -> atur kreativitas (0.1 = konsisten, 1.2 = liar), default 0.8
  /len <angka>    -> atur panjang balasan (jumlah karakter), default 300
  /exit /quit     -> keluar
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

MODEL_PATH = "bit_model.pt"
BLOCK_SIZE = 2048
N_EMBED = 2048
N_HEAD = 16
N_LAYER = 24
DROPOUT = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Arsitektur (harus sama persis dengan bit_train.py) ----------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class SwiGLU(nn.Module):
    def forward(self, x):
        return F.silu(x[..., :x.shape[-1]//2]) * x[..., x.shape[-1]//2:]

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        hidden_dim = int(4 * n_embed * 2 / 3)
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.act = SwiGLU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.w3(self.act(torch.cat([self.w1(x), self.w2(x)], dim=-1))))


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BitModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model '{MODEL_PATH}' belum ada.")
        print("Latih dulu modelnya dengan: python bit_train.py <data.txt>")
        sys.exit(1)

    print("Memuat BIT...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]

    model = BitModel(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("BIT siap. Model kecil ini murni hasil training dari data kamu sendiri.")
    print("Perintah: /temp <angka>  /len <angka>  /lang <bahasa>  /exit\n")

    temperature = 0.8
    gen_length = 300
    system_context = "Respond in the language requested by the user."
    unknown_char = " "  # fallback untuk karakter yang tidak ada di vocabulary

    while True:
        try:
            user_input = input("Kamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            print("Sampai jumpa!")
            break
        if user_input.lower().startswith("/temp"):
            try:
                temperature = float(user_input.split()[1])
                print(f"(temperature diatur ke {temperature})\n")
            except (IndexError, ValueError):
                print("Contoh: /temp 0.8\n")
            continue
        if user_input.lower().startswith("/len"):
            try:
                gen_length = int(user_input.split()[1])
                print(f"(panjang balasan diatur ke {gen_length} karakter)\n")
            except (IndexError, ValueError):
                print("Contoh: /len 300\n")
            continue
        if user_input.lower().startswith("/lang"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                system_context = f"Respond in {parts[1]}."
                print(f"(Bahasa diatur ke: {parts[1]})\n")
            else:
                print("Contoh: /lang Indonesia\n")
            continue

        # Encode input, prepended context
        prompt = f"{system_context}\nKamu: {user_input}\nBIT:"
        try:
            idx = torch.tensor(
                [[stoi.get(c, stoi.get(unknown_char, 0)) for c in prompt]],
                dtype=torch.long,
            ).to(device)
        except Exception as e:
            print(f"[Error encoding input]: {e}\n")
            continue

        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=gen_length, temperature=temperature)

        generated = "".join(itos[i] for i in out[0].tolist())
        reply = generated[len(prompt):]

        print(f"\nBIT: {reply}\n")


if __name__ == "__main__":
    main()
