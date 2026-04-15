"""
microgpt in PyTorch, MoE scaffold built on top of the baseline GPT.

This file intentionally stays close to `Basic GPT.py` so it can be used as a
drop-in experimental variant. The MoE pieces here are a skeleton:
- a token router
- per-token top-k expert selection
- optional shared expert
- auxiliary routing statistics for load balancing

The implementation is functional, but the routing / dispatch path is written in
clear PyTorch rather than optimized sparse kernels. That makes it easier to
iterate on the architecture first, then replace the hot path later.
"""

import math
from dataclasses import dataclass

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
    """Swap halves and negate."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply RoPE to x of shape (B, H, T, hd). Broadcasts over H."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
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
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
        )
        cos, sin = precompute_rope(self.head_dim, block_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

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
        attn = attn.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, embd_dims, hidden_mult=4):
        super().__init__()
        hidden_dims = hidden_mult * embd_dims
        self.fc1 = nn.Linear(embd_dims, hidden_dims, bias=False)
        self.fc2 = nn.Linear(hidden_dims, embd_dims, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ExpertMLP(nn.Module):
    """Single feed-forward expert."""

    def __init__(self, embd_dims, hidden_mult=4):
        super().__init__()
        self.ffn = MLP(embd_dims, hidden_mult=hidden_mult)

    def forward(self, x):
        return self.ffn(x)


@dataclass
class RouterOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    topk_idx: torch.Tensor
    topk_probs: torch.Tensor


class Router(nn.Module):
    """Token router for top-k expert selection."""

    def __init__(self, embd_dims, n_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.proj = nn.Linear(embd_dims, n_experts, bias=False)

    def forward(self, x):
        # TODO block 1: compute router logits from the token representations.
        scores = self.proj(x)

        # TODO block 2: turn logits into routing probabilities.
        probs = F.softmax(scores, dim=-1)

        # TODO block 3: select the top-k experts for each token.
        topk_probs, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)

        # TODO block 4: return a RouterOutput with all routing artifacts.
        return RouterOutput(scores, probs, topk_idx, topk_probs)


class MoEFeedForward(nn.Module):
    """Sparse-ish MoE feed-forward block.

    This is a readable implementation first:
    - route each token to top-k experts
    - evaluate the selected experts
    - mix their outputs by the routing weights
    - optionally add a shared expert

    It is not yet optimized for large-scale training.
    """

    def __init__(
        self,
        embd_dims,
        n_experts,
        top_k=2,
        hidden_mult=4,
        use_shared_expert=True,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert

        self.router = Router(embd_dims, n_experts, top_k=top_k)
        self.experts = nn.ModuleList(
            [ExpertMLP(embd_dims, hidden_mult=hidden_mult) for _ in range(n_experts)]
        )
        self.shared_expert = ExpertMLP(embd_dims, hidden_mult=hidden_mult)

    def forward(self, x, return_aux=False):
        # TODO block 1: route tokens to experts.

        router_out = self.router(x)
        scores, probs, topk_idx, topk_probs = (
            router_out.logits,
            router_out.probs,
            router_out.topk_idx,
            router_out.topk_probs,
        )

        # TODO block 2: run the selected experts on their assigned tokens.

        B, T, D = x.shape
        N = B * T
        K = self.top_k

        x_flat = x.reshape(N, D)
        idx_flat = topk_idx.reshape(-1)      # (N*K,)
        prob_flat = topk_probs.reshape(-1)    # (N*K,)
        tok_flat = torch.arange(N, device=x.device).repeat_interleave(K)
        order = torch.argsort(idx_flat)

        idx_flat = idx_flat[order]
        prob_flat = prob_flat[order]
        tok_flat = tok_flat[order]
        x_sel = x_flat[tok_flat]

        out_flat = x_flat.new_zeros(N, D)

        counts = torch.bincount(idx_flat, minlength=self.n_experts)
        start = 0
        for e, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            end = start + c
            y = self.experts[e](x_sel[start:end])   # one batched call per expert
            out_flat.index_add_(
                0,
                tok_flat[start:end],
                y * prob_flat[start:end].unsqueeze(-1),
            )
            start = end

        output = out_flat.view(B, T, D)

        # TODO block 4: optionally add the shared expert output.

        if self.use_shared_expert:
            output += self.shared_expert(x)

        aux = router_out

        if return_aux:
            return output, aux
        else:
            return output

    def aux_loss(self, router_probs):
        """Load-balancing loss helper.

        The markdown sketch tracks two expert-usage signals:
        - importance: total router probability mass per expert
        - load: how many tokens are assigned to each expert

        A common penalty combines the coefficient of variation of both.
        """
        # TODO block 1: compute f_i
        _, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        f = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=(0, 1, 2)).float()
        f = f / f.sum()

        # TODO block 2: compute p_i
        p = router_probs.sum(dim=(0, 1))
        p = p / p.sum()

        # TODO block 3: combine importance and load into a scalar balancing loss.
        return self.n_experts * (f * p).sum()


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


class MoEBlock(nn.Module):
    """Transformer block with MoE feed-forward instead of a dense MLP."""

    def __init__(
        self,
        embd_dims,
        n_head,
        block_size,
        n_experts,
        top_k=2,
        hidden_mult=4,
        use_shared_expert=True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(embd_dims)
        self.attn = CausalAttention(embd_dims, n_head, block_size)
        self.norm2 = RMSNorm(embd_dims)
        self.moe = MoEFeedForward(
            embd_dims,
            n_experts=n_experts,
            top_k=top_k,
            hidden_mult=hidden_mult,
            use_shared_expert=use_shared_expert,
        )

    def forward(self, x, return_aux=False):
        x = x + self.attn(self.norm1(x))

        # TODO block 1: normalize the residual stream before the MoE path.
        nx = self.norm1(x)
        # TODO block 2: send the normalized activations into the MoE feed-forward.
        if return_aux:
            moe_out, aux = self.moe(nx, return_aux=True)
        else:
            moe_out = self.moe(nx)
        # TODO block 3: add the MoE output back into the residual stream.
        x += moe_out
        if return_aux:
            return x, aux
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embd_dims, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)
        self.blocks = nn.ModuleList(
            [Block(embd_dims, n_head, block_size) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(embd_dims, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.wte(idx)
        x = self.norm0(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.8):
        """Autoregressive generation, token by token."""
        idx = start_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)
        return idx


class MoEGPT(nn.Module):
    """GPT variant with MoE feed-forward blocks."""

    def __init__(
        self,
        vocab_size,
        embd_dims,
        n_head,
        n_layer,
        block_size,
        n_experts=4,
        top_k=2,
        hidden_mult=4,
        use_shared_expert=True,
    ):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)
        self.blocks = nn.ModuleList(
            [
                MoEBlock(
                    embd_dims,
                    n_head,
                    block_size,
                    n_experts=n_experts,
                    top_k=top_k,
                    hidden_mult=hidden_mult,
                    use_shared_expert=use_shared_expert,
                )
                for _ in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(embd_dims, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, return_aux=False):
        x = self.norm0(self.wte(idx))
        if return_aux:
            total_aux = x.new_zeros(())
            for block in self.blocks:
                x, aux = block(x, return_aux=True)
                total_aux = total_aux + block.moe.aux_loss(aux.probs)
            return self.lm_head(x), total_aux

        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.8):
        idx = start_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)
        return idx


if __name__ == "__main__":
    print("Use this module as a GPT/MoE scaffold, or import MoEGPT into a run script.")
