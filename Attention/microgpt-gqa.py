"""
microgpt in PyTorch — GPU training on WikiText-103.
  RMSNorm, causal grouped-query attention (GQA), MLP with ReLU, Adam.

Requirements: pip install torch datasets tiktoken matplotlib
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- DataLoader ---
class TextDataset(torch.utils.data.Dataset):
    """Non-overlapping chunks of block_size from a flat token array."""
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
        self.n = (len(tokens) - 1) // block_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.block_size
        x = self.tokens[i : i + self.block_size]
        y = self.tokens[i + 1 : i + self.block_size + 1]
        return x, y


# --- RoPE ---
def precompute_rope(head_dim, max_seq_len, theta=10000.0):
    """Precompute cos/sin tables for rotary position embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)                  # (T, hd/2)
    cos = angles.cos().repeat(1, 2)                  # (T, hd)
    sin = angles.sin().repeat(1, 2)                  # (T, hd)
    return cos, sin


def rotate_half(x):
    """Swap halves and negate: [x0..x_{d/2-1}, x_{d/2}..x_{d-1}] → [-x_{d/2}.., x0..]"""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply RoPE to x of shape (B, H, T, hd). Broadcasts over H."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)          # (1, 1, T, hd)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# --- Model ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalAttention(nn.Module):
    def __init__(self, embd_dims, n_qhead, n_kvhead, block_size):
        super().__init__()
        self.n_qhead = n_qhead
        self.n_kvhead = n_kvhead
        self.head_dim = embd_dims // n_qhead
        self.wq = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wk = nn.Linear(embd_dims, self.n_kvhead * self.head_dim, bias=False)      # GQA: n_kvhead KV heads
        self.wv = nn.Linear(embd_dims, self.n_kvhead * self.head_dim, bias=False)
        self.wo = nn.Linear(embd_dims, embd_dims, bias=False)
        self.register_buffer('mask',
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))
        cos, sin = precompute_rope(self.head_dim, block_size)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, D = x.shape
        hd = self.head_dim
        QH = self.n_qhead
        KVH = self.n_kvhead

        q = self.wq(x).view(B, T, QH, hd).transpose(1, 2)           # (B, QH, T, hd)
        k = self.wk(x).view(B, T, KVH, hd).transpose(1, 2)          # (B, KVH, T, hd)
        v = self.wv(x).view(B, T, KVH, hd).transpose(1, 2)          # (B, KVH, T, hd)

        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)

        k = k.repeat_interleave(QH // KVH, dim=1)
        v = v.repeat_interleave(QH // KVH, dim=1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)            # (B, QH, T, T)
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, embd_dims):
        super().__init__()
        self.fc1 = nn.Linear(embd_dims, 4 * embd_dims, bias=False)
        self.fc2 = nn.Linear(4 * embd_dims, embd_dims, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, embd_dims, n_qhead, n_kvhead, block_size):
        super().__init__()
        self.norm1 = RMSNorm(embd_dims)
        self.attn = CausalAttention(embd_dims, n_qhead, n_kvhead, block_size)
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embd_dims, n_qhead, n_kvhead, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)
        self.blocks = nn.ModuleList([
            Block(embd_dims, n_qhead, n_kvhead, block_size) for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(embd_dims, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        x = self.norm0(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.8):
        """Autoregressive generation, token by token."""
        idx = start_ids  # (1, T_start)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]   # last position
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)
        return idx


if __name__ == '__main__':
    print("Use run_attention_comparison.py to train and evaluate.")
