"""
microgpt in PyTorch — GPU training on WikiText-103.
  RMSNorm, causal multi-latent attention (MLA), MLP with ReLU, Adam.

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
    def __init__(self, embd_dims, n_head, latent_dims, rope_dims, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = embd_dims // n_head
        self.rope_dim = rope_dims

        self.wqdown = nn.Linear(embd_dims, latent_dims, bias=False)
        self.wqup = nn.Linear(latent_dims, embd_dims, bias=False)
        self.wqr = nn.Linear(latent_dims, n_head * rope_dims, bias=False)

        self.wkvdown = nn.Linear(embd_dims, latent_dims, bias=False)
        self.wkup = nn.Linear(latent_dims, embd_dims, bias=False)
        self.wvup = nn.Linear(latent_dims, embd_dims, bias=False)
        self.wkr = nn.Linear(embd_dims, rope_dims, bias=False)

        self.wo = nn.Linear(embd_dims, embd_dims, bias=False)
        self.register_buffer('mask',
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))
        cos, sin = precompute_rope(rope_dims, block_size)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, D = x.shape
        hd = self.head_dim
        rd = self.rope_dim
        H = self.n_head

        ql = self.wqdown(x)
        q = self.wqup(ql).view(B, T, H, hd).transpose(1, 2)           # (B, H, T, hd)
        qr = apply_rotary_emb(self.wqr(ql).view(B, T, H, rd).transpose(1, 2), self.rope_cos, self.rope_sin)  # (B, H, T, rd)

        ckv = self.wkvdown(x)
        k = self.wkup(ckv).view(B, T, H, hd).transpose(1, 2)        # (B, H, T, hd)
        v = self.wvup(ckv).view(B, T, H, hd).transpose(1, 2)        # (B, H, T, hd)
        kr = apply_rotary_emb(self.wkr(x).unsqueeze(1), self.rope_cos, self.rope_sin)  # (B, 1, T, rd)

        attn = ((q @ k.transpose(-2, -1)) + (qr @ kr.transpose(-2, -1))) / math.sqrt(hd + rd)  # (B, H, T, T)
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
    def __init__(self, embd_dims, n_head, latent_dims, rope_dims, block_size):
        super().__init__()
        self.norm1 = RMSNorm(embd_dims)
        self.attn = CausalAttention(embd_dims, n_head, latent_dims, rope_dims, block_size)
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embd_dims, n_head, latent_dims, rope_dims, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)
        self.blocks = nn.ModuleList([
            Block(embd_dims, n_head, latent_dims, rope_dims, block_size) for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(embd_dims, vocab_size, bias=False)
        self.apply(self._init_weights)
        # Fix variance for factored projections: with W_down at std=0.02,
        # W_up needs std=1/√fan_in so the pair matches a single layer's output scale.
        for block in self.blocks:
            a = block.attn
            nn.init.normal_(a.wqup.weight, std=1.0 / math.sqrt(a.wqup.in_features))
            nn.init.normal_(a.wkup.weight, std=1.0 / math.sqrt(a.wkup.in_features))
            nn.init.normal_(a.wvup.weight, std=1.0 / math.sqrt(a.wvup.in_features))

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
        """MLA inference: absorb up-projections, cache only compressed ckv."""
        B, T_start = start_ids.shape
        device = start_ids.device
        idx = start_ids

        # ── 1. Precompute absorbed matrices per layer ──
        # Scores (row-vector form, head h):
        #   q_h @ k_h^T = (ql @ Wqup_h^T) @ (ckv @ Wkup_h^T)^T
        #               = ql @ (Wqup_h^T @ Wkup_h) @ ckv^T
        #               = ql @ W_qk_h @ ckv^T
        # Output:
        #   sum_h (attn_h @ ckv @ Wvup_h^T) @ Wo_h^T
        #       = sum_h attn_h @ ckv @ (Wvup_h^T @ Wo_h^T)
        #       = sum_h attn_h @ ckv @ W_vo_h
        absorbed_qk = []
        absorbed_vo = []
        for block in self.blocks:
            a = block.attn
            hd = a.head_dim
            H = a.n_head

            wqup = a.wqup.weight.view(H, hd, -1)             # (H, hd, latent)
            wkup = a.wkup.weight.view(H, hd, -1)             # (H, hd, latent)
            absorbed_qk.append(wqup.transpose(-1, -2) @ wkup) # (H, latent, latent)

            wvup = a.wvup.weight.view(H, hd, -1)             # (H, hd, latent)
            wo = a.wo.weight.T.contiguous().view(H, hd, -1)  # (H, hd, embd)
            absorbed_vo.append(wvup.transpose(-1, -2) @ wo)   # (H, latent, embd)
        # ── 2. Prefill: run prompt tokens, build ckv cache ──
        kv_cache = []  # per layer: (B, T, latent_dims)
        kr_cache = []
        T = T_start
        x = self.wte(idx)
        x = self.norm0(x)

        for i, block in enumerate(self.blocks):
            a = block.attn
            hd = a.head_dim
            rd = a.rope_dim
            H = a.n_head
            nx = block.norm1(x)

            ql = a.wqdown(nx)                                                                    # (B, T, latent)
            qr = apply_rotary_emb(a.wqr(ql).view(B, T, H, rd).transpose(1, 2), a.rope_cos, a.rope_sin)  # (B, H, T, rd)
            ckv = a.wkvdown(nx)                                                                  # (B, T, latent)
            kr = apply_rotary_emb(a.wkr(nx).unsqueeze(1), a.rope_cos, a.rope_sin)               # (B, 1, T, rd)

            kv_cache.append(ckv)
            kr_cache.append(kr)

            W_qk = absorbed_qk[i]
            W_vo = absorbed_vo[i]

            # scores = ql @ W_qk @ ckv^T + qr @ kr^T  → (B, H, T, T)
            content_scores = ql.unsqueeze(1) @ W_qk @ ckv.unsqueeze(1).transpose(-1, -2)
            rope_scores = qr @ kr.transpose(-1, -2)
            scores = (content_scores + rope_scores) / math.sqrt(hd + rd)
            scores = scores.masked_fill(a.mask[:T, :T], float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = (attn @ ckv.unsqueeze(1) @ W_vo).sum(dim=1)

            x = x + out
            x = x + block.mlp(block.norm2(x))

        # ── 3. Decode: one token at a time ──
        for step in range(max_new_tokens):
            pos = T_start + step
            x_new = self.wte(idx[:, -1:])
            x_new = self.norm0(x_new)  # (B, 1, embd_dims)

            for i, block in enumerate(self.blocks):
                a = block.attn
                hd = a.head_dim
                rd = a.rope_dim
                H = a.n_head
                nx = block.norm1(x_new)
                cos_pos = a.rope_cos[pos:pos+1]
                sin_pos = a.rope_sin[pos:pos+1]

                ql = a.wqdown(nx)                                                          # (B, 1, latent)
                qr = apply_rotary_emb(a.wqr(ql).view(B, 1, H, rd).transpose(1, 2), cos_pos, sin_pos)  # (B, H, 1, rd)

                ckv = a.wkvdown(nx)                                                        # (B, 1, latent)
                kv_cache[i] = torch.cat([kv_cache[i], ckv], dim=1)
                ckv_all = kv_cache[i]                                                      # (B, T_cur, latent)
                kr = apply_rotary_emb(a.wkr(nx).unsqueeze(1), cos_pos, sin_pos)            # (B, 1, 1, rd)
                kr_cache[i] = torch.cat([kr_cache[i], kr], dim=2)
                kr_all = kr_cache[i]

                W_qk = absorbed_qk[i]
                W_vo = absorbed_vo[i]

                # scores → (B, H, 1, T_cur), no causal mask needed (single new token)
                content_scores = ql.unsqueeze(1) @ W_qk @ ckv_all.unsqueeze(1).transpose(-1, -2)
                rope_scores = qr @ kr_all.transpose(-1, -2)
                scores = (content_scores + rope_scores) / math.sqrt(hd + rd)
                attn = F.softmax(scores, dim=-1)
                out = (attn @ ckv_all.unsqueeze(1) @ W_vo).sum(dim=1)

                x_new = x_new + out
                x_new = x_new + block.mlp(block.norm2(x_new))

            logits = self.lm_head(x_new)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)

        return idx


if __name__ == '__main__':
    print("Use run_attention_comparison.py to train and evaluate.")