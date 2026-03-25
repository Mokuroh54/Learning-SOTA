"""
microgpt in PyTorch — GPU training on WikiText-103.
  RMSNorm, causal multi-head attention (MHA), MLP with ReLU, Adam.

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

        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)           # (B, H, T, hd)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)           # (B, H, T, hd)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)           # (B, H, T, hd)

        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)            # (B, H, T, T)
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
    def __init__(self, embd_dims, n_head, block_size):
        super().__init__()
        self.norm1 = RMSNorm(embd_dims)
        self.attn = CausalAttention(embd_dims, n_head, block_size)
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --- Attention Residuals ---

class AttentionResidual(nn.Module):
    """Per-layer attention residual: learns how much to weight the residual
    stream vs the layer output using a learned query vector and attention."""

    def __init__(self, embd_dims):
        super().__init__()
        # Query vector initialized to 0 so initial attention weights are equal
        self.q = nn.Parameter(torch.zeros(embd_dims))
        self.key_norm = RMSNorm(embd_dims)

    def forward(self, residual, layer_out):
        """
        residual:  x_{i-1}          (B, T, D)
        layer_out: f_i(x_{i-1})     (B, T, D)
        returns:   weighted combination of residual and layer_out
        """

        scores_r = (self.q * self.key_norm(residual)).sum(dim=-1, keepdim=True)
        scores_o = (self.q * self.key_norm(layer_out)).sum(dim=-1, keepdim=True)

        scores = F.softmax(torch.cat([scores_r, scores_o], dim=-1), dim=-1)

        return scores[:, :, 0] * residual + scores[:, :, 1] * layer_out


class ARLayer(nn.Module):
    """Single transformer layer with a learned query vector for inter-block attention."""

    def __init__(self, embd_dims, n_head, block_size):
        super().__init__()
        self.norm1 = RMSNorm(embd_dims)
        self.attn = CausalAttention(embd_dims, n_head, block_size)
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)
        # Per-layer query vector for depth-wise attention, init to 0
        self.q = nn.Parameter(torch.zeros(embd_dims))

    def forward(self, x):
        """Raw sublayer forward — no residual connections.
        The depth-wise attention residual is handled externally by ARBlock."""
        x = self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x


def attn_with_stats(q, k, v, key_norm):
    """Compute attention and return output with softmax statistics.

    Args:
        q: learned query vector                    (D,)
        k: key tensor (will be RMSNorm'd)          (B, T, D) or (num_keys, B, T, D)
        v: value tensor (same shape as k)
        key_norm: RMSNorm module
    Returns:
        out:  weighted output                      (B, T, D)
        m:    max score                            (B, T, 1)
        l:    sum of exp(score - m)                (B, T, 1)
    """
    
    k_norm = key_norm(k)
    score = (q * k_norm).sum(dim=-1, keepdim=True)
    max_score = (torch.max(score, dim=0).values)
    softmax_num = torch.exp(score - max_score)
    softmax_denom = softmax_num.sum(dim=0, keepdim=True)
    return (((softmax_num / softmax_denom) * v).sum(dim=0), max_score.squeeze(dim=0), softmax_denom.squeeze(dim=0))


def online_softmax_merge(o1, m1, l1, o2, m2, l2):
    """Merge two partial softmax results using online softmax.

    Args:
        o1, m1, l1: output, max, sum-of-exps from Phase 1   (B, T, D), (B, T, 1), (B, T, 1)
        o2, m2, l2: output, max, sum-of-exps from Phase 2   (B, T, D), (B, T, 1), (B, T, 1)
    Returns:
        merged output (B, T, D)
    """
    
    m = max(m1, m2)
    o1 *= torch.exp(m1 - m)
    o2 *= torch.exp(m2 - m)
    l1 *= torch.exp(m1 - m)
    l2 *= torch.exp(m2 - m)
    return (o1 + o2) / (l1 + l2)


class ARBlock(nn.Module):
    """Block of layers with two-phase attention residuals.

    Phase 1 (parallel):  Batch all per-layer queries in this block against
                         previous completed block summaries. Returns partial
                         softmax stats (output, max, lse) per layer.
    Phase 2 (sequential): Run each layer, accumulating the intra-block partial
                          sum. For each layer, compute single-key attention
                          against the partial sum, then merge with Phase 1
                          results via online softmax.
    """

    def __init__(self, embd_dims, n_head, block_size, n_layers_in_block):
        super().__init__()
        self.layers = nn.ModuleList([
            ARLayer(embd_dims, n_head, block_size) for _ in range(n_layers_in_block)
        ])
        self.key_norm = RMSNorm(embd_dims)

    def forward(self, x, prev_block_summaries):
        """
        x:                    input to this block                    (B, T, D)
        prev_block_summaries: list of completed block sums           [(B, T, D), ...]

        Returns:
            x:          output of this block                         (B, T, D)
            block_sum:  sum of layer outputs in this block           (B, T, D)
        """

        k = torch.stack(prev_block_summaries, dim=0)
        inter_block = []
        for layer in self.layers:
            attn_out = attn_with_stats(layer.q, k, k, self.key_norm)
            inter_block.append(attn_out)
        
        intra_block = torch.zeros_like(x)
        for i, layer in enumerate(self.layers):
            if i > 0:
                attn_out = attn_with_stats(layer.q, intra_block.unsqueeze(0), intra_block.unsqueeze(0), self.key_norm)
                x = online_softmax_merge(*inter_block[i], *attn_out)
            else:
                o, m, l = inter_block[i]
                x = o / l
            
            intra_block += layer(x)
        
        return x, intra_block


class ARGPT(nn.Module):
    """GPT with block attention residuals. Layers are grouped into blocks
    of sqrt(L). b_0 = token embedding. Each subsequent block produces a
    block summary (sum of its layer outputs). Inter-block attention is
    handled inside ARBlock via the two-phase approach."""

    def __init__(self, vocab_size, embd_dims, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)

        # Group layers into blocks of ~sqrt(n_layer)
        self.layers_per_block = max(1, int(math.sqrt(n_layer)))
        n_blocks = math.ceil(n_layer / self.layers_per_block)

        self.blocks = nn.ModuleList()
        for b in range(n_blocks):
            n_in_block = min(self.layers_per_block, n_layer - b * self.layers_per_block)
            self.blocks.append(
                ARBlock(embd_dims, n_head, block_size, n_in_block)
            )

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

        b_0 = x
        block_summaries = [b_0]
        for block in self.blocks:
            x, block_sum = block.forward(x, block_summaries)
            block_summaries.append(block_sum)
        
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.8):
        idx = start_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)
        return idx


class GPT(nn.Module):
    def __init__(self, vocab_size, embd_dims, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)
        self.blocks = nn.ModuleList([
            Block(embd_dims, n_head, block_size) for _ in range(n_layer)
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
