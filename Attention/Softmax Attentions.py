"""
microgpt unified — all softmax attention variants in one file.

Supported attention types (pass as attn_type to GPT):
  'mha'  — Multi-Head Attention (N query heads, N KV heads)
  'mqa'  — Multi-Query Attention (N query heads, 1 shared KV head)
  'gqa'  — Grouped-Query Attention (N query heads, M KV heads)
  'mla'  — Multi-Latent Attention (factorized projections, shared KV compression)
  'dsa'  — DeepSeek Sparse Attention (MLA + Lightning Indexer for top-k selection)

Each attention variant is its own class. Block and GPT route to the
correct one based on attn_type, so the training/inference harness stays
the same regardless of which attention you pick.

Requirements: pip install torch datasets tiktoken matplotlib
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  Shared utilities
# ═══════════════════════════════════════════════════════════════════════

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


# --- DSA helpers ---
def fp8_ste(x):
    """Quantize to FP8 with straight-through estimator for gradients."""
    x_q = x.to(torch.float8_e4m3fn).to(x.dtype)
    return x + (x_q - x).detach()


def fwht(x):
    """Fast Walsh-Hadamard Transform along the last dimension (unnormalized).
    Last dimension must be a power of 2."""
    n = x.shape[-1]
    h = 1
    while h < n:
        xv = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        xv[..., 0, :].add_(xv[..., 1, :])
        xv[..., 1, :].mul_(-2).add_(xv[..., 0, :])
        h *= 2
    return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MLP(nn.Module):
    def __init__(self, embd_dims):
        super().__init__()
        self.fc1 = nn.Linear(embd_dims, 4 * embd_dims, bias=False)
        self.fc2 = nn.Linear(4 * embd_dims, embd_dims, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ═══════════════════════════════════════════════════════════════════════
#  Attention Variants
# ═══════════════════════════════════════════════════════════════════════

class MHAttention(nn.Module):
    """Multi-Head Attention — N query heads, N KV heads."""
    def __init__(self, embd_dims, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = embd_dims // n_head
        self.wq = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wk = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wv = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wo = nn.Linear(embd_dims, embd_dims, bias=False)
        self.register_buffer('mask',
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))
        cos, sin = precompute_rope(self.head_dim, block_size)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, D = x.shape
        hd = self.head_dim
        H = self.n_head

        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)

        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class MQAttention(nn.Module):
    """Multi-Query Attention — N query heads, 1 shared KV head."""
    def __init__(self, embd_dims, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = embd_dims // n_head
        self.wq = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wk = nn.Linear(embd_dims, self.head_dim, bias=False)
        self.wv = nn.Linear(embd_dims, self.head_dim, bias=False)
        self.wo = nn.Linear(embd_dims, embd_dims, bias=False)
        self.register_buffer('mask',
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))
        cos, sin = precompute_rope(self.head_dim, block_size)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, D = x.shape
        hd = self.head_dim
        H = self.n_head

        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, 1, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, 1, hd).transpose(1, 2)

        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class GQAttention(nn.Module):
    """Grouped-Query Attention — N query heads, M KV heads (M divides N)."""
    def __init__(self, embd_dims, n_qhead, n_kvhead, block_size):
        super().__init__()
        self.n_qhead = n_qhead
        self.n_kvhead = n_kvhead
        self.head_dim = embd_dims // n_qhead
        self.wq = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wk = nn.Linear(embd_dims, self.n_kvhead * self.head_dim, bias=False)
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

        q = self.wq(x).view(B, T, QH, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, KVH, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, KVH, hd).transpose(1, 2)

        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)

        k = k.repeat_interleave(QH // KVH, dim=1)
        v = v.repeat_interleave(QH // KVH, dim=1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class MLAttention(nn.Module):
    """Multi-Latent Attention — factorized projections with shared KV compression."""
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
        q = self.wqup(ql).view(B, T, H, hd).transpose(1, 2)
        qr = apply_rotary_emb(self.wqr(ql).view(B, T, H, rd).transpose(1, 2), self.rope_cos, self.rope_sin)

        ckv = self.wkvdown(x)
        k = self.wkup(ckv).view(B, T, H, hd).transpose(1, 2)
        v = self.wvup(ckv).view(B, T, H, hd).transpose(1, 2)
        kr = apply_rotary_emb(self.wkr(x).unsqueeze(1), self.rope_cos, self.rope_sin)

        attn = ((q @ k.transpose(-2, -1)) + (qr @ kr.transpose(-2, -1))) / math.sqrt(hd + rd)
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class DSAttention(nn.Module):
    """DeepSeek Sparse Attention — MLA with optional top-k sparse selection."""
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

    def forward(self, x, topk_idx=None, return_attn=False):
        B, T, D = x.shape
        hd = self.head_dim
        rd = self.rope_dim
        H = self.n_head

        ql = self.wqdown(x)
        q = self.wqup(ql).view(B, T, H, hd).transpose(1, 2)
        qr = apply_rotary_emb(self.wqr(ql).view(B, T, H, rd).transpose(1, 2), self.rope_cos, self.rope_sin)

        ckv = self.wkvdown(x)
        k = self.wkup(ckv).view(B, T, H, hd).transpose(1, 2)
        v = self.wvup(ckv).view(B, T, H, hd).transpose(1, 2)
        kr = apply_rotary_emb(self.wkr(x).unsqueeze(1), self.rope_cos, self.rope_sin)

        if topk_idx is None:
            attn = ((q @ k.transpose(-2, -1)) + (qr @ kr.transpose(-2, -1))) / math.sqrt(hd + rd)
            attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, T, D)
            out = self.wo(out)
            return (out, attn) if return_attn else out

        # Sparse attention: gather selected keys, values, rope keys
        K = topk_idx.shape[-1]
        flat_idx = topk_idx.reshape(B, -1)
        idx_hd = flat_idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, hd)
        k_sel = k.gather(2, idx_hd).reshape(B, H, T, K, hd)
        v_sel = v.gather(2, idx_hd).reshape(B, H, T, K, hd)

        idx_rd = flat_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, rd)
        kr_sel = kr.gather(2, idx_rd).reshape(B, 1, T, K, rd)

        scores = (torch.einsum('bhtd,bhtkd->bhtk', q, k_sel)
                + torch.einsum('bhtd,bhtkd->bhtk', qr, kr_sel.expand(-1, H, -1, -1, -1)))
        scores = scores / math.sqrt(hd + rd)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhtk,bhtkd->bhtd', attn, v_sel)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class LightningIndexer(nn.Module):
    """Lightning Indexer for DSA — FWHT + FP8 STE scoring for top-k key selection."""
    def __init__(self, embd_dims, n_head, latent_dims, rope_dims, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = embd_dims // n_head
        self.rope_dim = rope_dims

        self.wqdown = nn.Linear(embd_dims, latent_dims, bias=False)
        self.wqup = nn.Linear(latent_dims, embd_dims, bias=False)
        self.wqr = nn.Linear(latent_dims, n_head * self.rope_dim, bias=False)

        self.wkdown = nn.Linear(embd_dims, latent_dims, bias=False)
        self.wkup = nn.Linear(latent_dims, self.head_dim, bias=False)
        self.wkr = nn.Linear(embd_dims, self.rope_dim, bias=False)
        self.wh = nn.Linear(latent_dims, n_head, bias=False)

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
        q = self.wqup(ql).view(B, T, H, hd).transpose(1, 2)
        qr = apply_rotary_emb(self.wqr(ql).view(B, T, H, rd).transpose(1, 2), self.rope_cos, self.rope_sin)

        ckv = self.wkdown(x)
        k = self.wkup(ckv).view(B, T, 1, hd).transpose(1, 2)
        kr = apply_rotary_emb(self.wkr(x).unsqueeze(1), self.rope_cos, self.rope_sin)

        q = fp8_ste(fwht(q.contiguous()) / math.sqrt(hd))
        k = fp8_ste(fwht(k.contiguous()) / math.sqrt(hd))
        qr = fp8_ste(fwht(qr.contiguous()) / math.sqrt(rd))
        kr = fp8_ste(fwht(kr.contiguous()) / math.sqrt(rd))

        lis = ((q @ k.transpose(-2, -1)) + (qr @ kr.transpose(-2, -1)))
        lis = lis.masked_fill(self.mask[:T, :T], float('-inf'))
        lis = F.relu(lis)

        hw = self.wh(ql)
        hw = F.sigmoid(hw)
        hw = hw.transpose(1, 2).unsqueeze(-1)

        return (hw * lis).sum(dim=1)


# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

ATTN_REGISTRY = {
    'mha': MHAttention,
    'mqa': MQAttention,
    'gqa': GQAttention,
    'mla': MLAttention,
    'dsa': DSAttention,
}

# Types with MLA-style factored projections (need special weight init)
_MLA_TYPES = {'mla', 'dsa'}


# ═══════════════════════════════════════════════════════════════════════
#  Unified Block & GPT
# ═══════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    def __init__(self, attn_type, embd_dims, block_size, *,
                 n_head=None, n_qhead=None, n_kvhead=None,
                 latent_dims=None, rope_dims=None, top_k=None):
        super().__init__()
        self.attn_type = attn_type
        self.norm1 = RMSNorm(embd_dims)
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)

        if attn_type == 'mha':
            self.attn = MHAttention(embd_dims, n_head, block_size)
        elif attn_type == 'mqa':
            self.attn = MQAttention(embd_dims, n_head, block_size)
        elif attn_type == 'gqa':
            self.attn = GQAttention(embd_dims, n_qhead, n_kvhead, block_size)
        elif attn_type == 'mla':
            self.attn = MLAttention(embd_dims, n_head, latent_dims, rope_dims, block_size)
        elif attn_type == 'dsa':
            self.top_k = top_k
            self.lindexer = LightningIndexer(embd_dims, n_head, latent_dims, rope_dims, block_size)
            self.attn = DSAttention(embd_dims, n_head, latent_dims, rope_dims, block_size)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type!r}. "
                             f"Valid types: {list(ATTN_REGISTRY.keys())}")

    def forward(self, x, warmup=False):
        if self.attn_type == 'dsa':
            lis = self.lindexer(x)
            nx = self.norm1(x)
            T = x.shape[1]

            if warmup:
                out, p = self.attn(nx, topk_idx=None, return_attn=True)
                x = x + out.detach()
                p = p.detach().sum(dim=1)
                p = p / p.sum(dim=-1, keepdim=True)
                lis = lis.masked_fill(self.attn.mask[:T, :T], float('-inf'))
                aux_loss = F.kl_div(F.log_softmax(lis, dim=-1), p, reduction='batchmean')
            else:
                _, topk_idx = torch.topk(lis, min(self.top_k, T), dim=-1)
                x = x + self.attn(nx, topk_idx)
                aux_loss = torch.tensor(0.0, device=x.device)

            x = x + self.mlp(self.norm2(x))
            return x, aux_loss
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class GPT(nn.Module):
    """
    Unified GPT that accepts any softmax attention type.

    Constructor kwargs by attn_type:
      mha:  n_head
      mqa:  n_head
      gqa:  n_qhead, n_kvhead
      mla:  n_head, latent_dims, rope_dims
      dsa:  n_head, latent_dims, rope_dims, top_k, warmup_steps
    """
    def __init__(self, attn_type, vocab_size, embd_dims, n_layer, block_size, *,
                 n_head=None, n_qhead=None, n_kvhead=None,
                 latent_dims=None, rope_dims=None,
                 top_k=64, warmup_steps=0):
        super().__init__()
        self.attn_type = attn_type
        self.block_size = block_size
        self.warmup_steps = warmup_steps if attn_type == 'dsa' else 0

        if attn_type == 'dsa':
            self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)

        block_kwargs = dict(
            attn_type=attn_type, embd_dims=embd_dims, block_size=block_size,
            n_head=n_head, n_qhead=n_qhead, n_kvhead=n_kvhead,
            latent_dims=latent_dims, rope_dims=rope_dims, top_k=top_k,
        )
        self.blocks = nn.ModuleList([Block(**block_kwargs) for _ in range(n_layer)])

        self.lm_head = nn.Linear(embd_dims, vocab_size, bias=False)
        self.apply(self._init_weights)

        # Fix variance for MLA/DSA factored projections
        if attn_type in _MLA_TYPES:
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
        is_dsa = self.attn_type == 'dsa'
        warmup = is_dsa and self.training and self._step < self.warmup_steps

        x = self.wte(idx)
        x = self.norm0(x)

        if is_dsa:
            total_aux = torch.tensor(0.0, device=idx.device)
            for block in self.blocks:
                x, aux = block(x, warmup=warmup)
                total_aux = total_aux + aux
            if self.training:
                self._step += 1
            logits = self.lm_head(x)
            return (logits.detach(), total_aux) if warmup else logits
        else:
            for block in self.blocks:
                x = block(x)
            return self.lm_head(x)

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.8):
        """Autoregressive generation, token by token."""
        idx = start_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            out = self(idx_cond)
            logits = out[0][:, -1, :] if isinstance(out, tuple) else out[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)
        return idx


if __name__ == '__main__':
    print("Unified microgpt — all softmax attention variants in one file.")
    print(f"Available attention types: {list(ATTN_REGISTRY.keys())}")
    print("\nExample usage:")
    print("  model = GPT('mha', vocab_size=50257, embd_dims=128, n_layer=4, block_size=256, n_head=4)")
    print("  model = GPT('mqa', vocab_size=50257, embd_dims=128, n_layer=4, block_size=256, n_head=4)")
    print("  model = GPT('gqa', vocab_size=50257, embd_dims=128, n_layer=4, block_size=256, n_qhead=8, n_kvhead=2)")
    print("  model = GPT('mla', vocab_size=50257, embd_dims=128, n_layer=4, block_size=256, n_head=4, latent_dims=64, rope_dims=32)")
    print("  model = GPT('dsa', vocab_size=50257, embd_dims=128, n_layer=4, block_size=256, n_head=4, latent_dims=64, rope_dims=32, top_k=64)")
