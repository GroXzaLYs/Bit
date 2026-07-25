#!/usr/bin/env python3
"""
BIT - Training Script
----------------------
Melatih neural network mini (transformer, dibuat dari nol) dari file teks
yang kamu sediakan. Model belajar level KARAKTER (bukan pakai model siapa pun).

Cara pakai:
  1. pip install torch
  2. Siapkan file teks training, misal data.txt (semakin banyak & relevan, semakin baik)
     - Untuk chat: kumpulkan contoh percakapan
     - Untuk coding: kumpulkan contoh kode Python
  3. Jalankan:
       python bit_train.py data.txt
  4. Model tersimpan sebagai bit_model.pt
  5. Chat dengan model: python bit_chat.py

Catatan jujur:
  Ini model KECIL yang belajar dari NOL (bobot acak di awal, tanpa model
  pretrained apa pun). Dengan data terbatas & training di CPU/laptop biasa,
  Bit akan belajar pola bahasa level dasar - bukan seperti ChatGPT/Claude.
  Semakin banyak & bagus data + semakin lama training, semakin pintar.
"""

import sys
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- Konfigurasi ----------------
BLOCK_SIZE = 2048     # panjang konteks yang lebih besar
BATCH_SIZE = 8        # dikurangi agar tidak OOM
N_EMBED = 2048        # ukuran embedding untuk ~1B
N_HEAD = 16           # jumlah attention head
N_LAYER = 24          # jumlah layer transformer
DROPOUT = 0.1
LEARNING_RATE = 1e-4  # learning rate disesuaikan
MAX_ITERS = 100000    # iterasi lebih banyak
EVAL_INTERVAL = 1000
MODEL_PATH = "bit_model.pt"

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Arsitektur Transformer (dari nol) ----------------
class Head(nn.Module):
    """Satu head self-attention."""
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
        # SwiGLU takes W1x and W2x, each size hidden_dim.
        # Concatenation makes it 2*hidden_dim.
        # Wait, my SwiGLU implementation above took the whole thing and split it?
        # Let's fix the logic to match the SwiGLU implementation:
        # SwiGLU expects W1x and W2x separately if using gated logic directly, 
        # but my SwiGLU class expects a combined tensor.
        
        # Let's re-align.
        
        # New approach:
        # hidden_dim is size of W1 and W2 output.
        # Concatenate W1 and W2 -> size 2 * hidden_dim.
        # SwiGLU splits this into two hidden_dim, and gates them.
        # Correct.
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
    """BIT - model bahasa mini, arsitektur transformer, dibuat & dilatih dari nol."""
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

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------- Training ----------------
def main():
    if len(sys.argv) < 2:
        print("Cara pakai: python bit_train.py <file_teks.txt>")
        sys.exit(1)

    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"File tidak ditemukan: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) < 2000:
        print("Peringatan: data teks sangat sedikit. Bit akan belajar sangat terbatas.")
        print("Semakin banyak teks (idealnya puluhan ribu kata+), semakin baik hasilnya.\n")

    # Vocabulary level karakter
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(50)
            for k in range(50):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    print(f"Perangkat: {device}")
    print(f"Ukuran vocabulary (karakter unik): {vocab_size}")
    print(f"Total karakter data: {len(text)}\n")
    print("Memulai training BIT dari nol (bobot acak, belum tahu apa-apa)...\n")

    model = BitModel(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Jumlah parameter model: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({
        "model_state": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }, MODEL_PATH)

    print(f"\nSelesai! Model BIT tersimpan di: {MODEL_PATH}")
    print("Sekarang jalankan: python bit_chat.py")


if __name__ == "__main__":
    main()
