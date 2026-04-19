"""
Baseline GPT plus an Engram scaffold.

This file stays close to the repo's existing GPT implementations so it can be
used as another experimental variant without refactoring the rest of the code.
The Engram path is intentionally incomplete: the TODO blocks mark the parts
that depend on your retrieval design rather than a fixed implementation choice.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 54

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
    angles = torch.outer(t, freqs)
    cos = angles.cos().repeat(1, 2)
    sin = angles.sin().repeat(1, 2)
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
        """Run the baseline GPT.

        Args:
            idx: Token ids with shape (B, T) and dtype torch.long. Each entry
                should be in [0, vocab_size). T may be smaller than
                self.block_size, but not larger.
        Returns:
            Logits with shape (B, T, vocab_size).
        """
        x = self.wte(idx)
        x = self.norm0(x)
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


# --- Prime utilities for Engram addressing ---
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _miller_rabin(n):
    """Deterministic primality test for any n < 3.3e24.

    Args:
        n: Non-negative Python int to test.
    Returns:
        True iff n is prime.

    Uses the fixed witness set [2, 3, ..., 37], which is proven to give no
    false positives for any value that fits in int64, so this is exact for
    every modulus we will ever want here.
    """
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Write n - 1 = 2^s * d with d odd.
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL_PRIMES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n):
    """Smallest prime p with p >= n.

    Args:
        n: Non-negative integer lower bound.
    Returns:
        Python int p, prime, satisfying p >= n.
    """
    if n <= 2:
        return 2
    candidate = n if n % 2 == 1 else n + 1
    while not _miller_rabin(candidate):
        candidate += 2
    return candidate


def next_primes(n, count):
    """The first `count` distinct primes with value >= n, ascending.

    Args:
        n: Non-negative integer lower bound for every returned prime.
        count: Number of primes to return.
    Returns:
        List of Python ints of length `count`, strictly increasing, all prime.

    Use this when you want independent moduli across hash heads while still
    pinning them to roughly the same memory budget.
    """
    primes = []
    lower = n
    for _ in range(count):
        p = next_prime(lower)
        primes.append(p)
        lower = p + 1
    return primes


# --- Engram scaffold ---
class EngramMemory(nn.Module):
    """Static lookup table addressed by multi-head n-gram hashing.

    Expected input:
        idx: Token ids with shape (B, T) and dtype torch.long. The same
            token-id tensor used by the embedding layer, not hidden states —
            addressing is driven by token identity and local token context,
            not by the contextual residual stream.
    """

    def __init__(self, embd_dims, memory_size, vocab_size, ngram=2, num_heads=1):
        super().__init__()
        self.memory_size = memory_size
        self.vocab_size = vocab_size
        self.ngram = ngram
        self.num_heads = num_heads

        g = torch.Generator()
        g.manual_seed(seed)

        max_long = torch.iinfo(torch.int64).max
        m_max = max_long // vocab_size
        half_bound = max(1, m_max // 2)

        # Per-(position, head) odd multipliers: shape (ngram, num_heads).
        # All entries are independent draws so no two heads share a mixing
        # pattern, preserving collision independence across heads.
        self.register_buffer(
            "m", 
            torch.randint(
                low=0,
                high=half_bound,
                size=(self.ngram, num_heads),
                dtype=torch.int64,
                generator=g,
            ) * 2 + 1
        )

        # Per-head prime moduli, each >= memory_size. Using distinct primes
        # across heads means the address distributions don't share factors,
        # so a collision in head i is uncorrelated with a collision in head j.
        primes = next_primes(memory_size, num_heads)
        self.register_buffer("M", torch.tensor(primes, dtype=torch.int64))

        # Table is sized to the largest per-head modulus so every head's
        # address space is reachable. This keeps the table shared across
        # heads; a per-head-slice layout with offsets would size it to
        # sum(primes) instead and is a block-1/block-2 decision.
        self.table = nn.Parameter(torch.empty(max(primes), embd_dims))
        nn.init.normal_(self.table, std=0.02)

    def compute_addresses(self, idx):
        """Map token ids to memory-table addresses.

        Args:
            idx: Integer token ids of shape (B, T). Each value identifies a
                vocabulary item from the tokenizer.
        Returns:
            Integer addresses that index into self.table. The trailing shape
            is up to you: (B, T) for a single slot per position, or
            (B, T, H) if you produce H parallel hash reads.
        """
        # Block 1 — multi-head n-gram hash.
        # Left-pad so every position sees a full ngram window, slide that
        # window across the sequence, then fold the per-position products
        # together under XOR (one mixing step per position in the ngram).
        # A final per-head modulus keeps each head's address stream inside
        # its own prime space.

        windows = F.pad(idx, (self.ngram - 1, 0)).unfold(dimension=1, size=self.ngram, step=1)
        products = (self.m * windows.unsqueeze(-1))
        ret = products[..., 0, :]
        for k in range(1, self.ngram):
            ret = ret ^ products[:, :, k, :]

        return ret.remainder(self.M)


    def forward(self, idx):
        """Look up memory rows for each token position.

        Args:
            idx: Token ids with shape (B, T) and dtype torch.long.
        Returns:
            Retrieved memory vectors with shape (B, T, D), where D is the
            embedding dimension used by the transformer residual stream.
        """
        addresses = self.compute_addresses(idx)

        # Block 2 — address lookup and head aggregation.
        # Gather one row per (position, head) from the shared table, then
        # collapse the head axis by summation. The sum is intentionally
        # parameter-free; all learned filtering of the retrieved signal
        # lives downstream in EngramFusion.

        return F.embedding(addresses, self.table).sum(dim=-2)


class EngramFusion(nn.Module):
    """Fuse retrieved memory back into the residual stream."""

    def __init__(self, embd_dims, kernel_size=3):
        super().__init__()
        self.query_norm = RMSNorm(embd_dims)
        self.key_norm = RMSNorm(embd_dims)
        self.value_norm = RMSNorm(embd_dims)
        self.wk = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wv = nn.Linear(embd_dims, embd_dims, bias=False)
        self.short_conv = nn.Conv1d(
            embd_dims,
            embd_dims,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embd_dims,
            bias=False,
        )

    def forward(self, x, retrieved):
        """Fuse retrieved memory into the residual stream.

        Args:
            x: Hidden states with shape (B, T, D). This is the current
                contextual representation for each token position.
            retrieved: Memory vectors with shape (B, T, D) aligned to the same
                token positions as x.
        Returns:
            An Engram residual update with shape (B, T, D).
        """
        # Block 3 — context-aware gate.
        # Score retrieval relevance by a per-position dot product between the
        # normalized hidden state and the normalized key-projected memory.
        # A signed-sqrt squash tames the dot-product magnitude before the
        # sigmoid so the gate stays in a usable range, and the gated scalar
        # then scales the value projection elementwise.

        scores = (self.query_norm(x) * self.key_norm(self.wk(retrieved))).sum(dim=-1)
        scores = scores.sign() * scores.abs().clamp_min(1e-6).sqrt()
        values = torch.sigmoid(scores / math.sqrt(x.shape[-1])).unsqueeze(-1) * (self.wv(retrieved))

        # Block 4 — short-conv mixing branch.
        # A depthwise Conv1d over the normalized gated values widens the
        # receptive field across neighboring positions; SiLU then gives a
        # nonlinear refinement which is added back as a residual on top of
        # the gated values. The transpose sandwich reshapes (B, T, D) into
        # the (B, C, L) layout Conv1d expects.

        return values + F.silu(self.short_conv(self.value_norm(values).transpose(-1, -2)).transpose(-1, -2))


class EngramLayer(nn.Module):
    """Lookup + fusion module inserted into the transformer stack."""

    def __init__(
        self,
        embd_dims,
        memory_size,
        vocab_size,
        ngram=2,
        kernel_size=3,
        engram_heads=1,
    ):
        super().__init__()
        self.memory = EngramMemory(
            embd_dims,
            memory_size,
            vocab_size,
            ngram=ngram,
            num_heads=engram_heads,
        )
        self.fusion = EngramFusion(embd_dims, kernel_size=kernel_size)

    def forward(self, idx, x):
        """Run lookup and fusion for one Engram insertion point.

        Args:
            idx: Token ids with shape (B, T) and dtype torch.long.
            x: Hidden states with shape (B, T, D).
        """
        retrieved = self.memory(idx)
        return self.fusion(x, retrieved)


class EngramBlock(nn.Module):
    """Transformer block with an extra Engram residual path."""

    def __init__(
        self,
        embd_dims,
        n_head,
        block_size,
        memory_size,
        vocab_size,
        use_engram=True,
        ngram=2,
        kernel_size=3,
        engram_heads=1,
    ):
        super().__init__()
        self.use_engram = use_engram
        self.norm1 = RMSNorm(embd_dims)
        self.attn = CausalAttention(embd_dims, n_head, block_size)
        self.engram = EngramLayer(
            embd_dims,
            memory_size,
            vocab_size=vocab_size,
            ngram=ngram,
            kernel_size=kernel_size,
            engram_heads=engram_heads,
        )
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)

    def forward(self, idx, x):
        """Apply attention, optional Engram update, and MLP.

        Args:
            idx: Token ids with shape (B, T) and dtype torch.long.
            x: Residual-stream activations with shape (B, T, D).
        Returns:
            Updated hidden states with shape (B, T, D).
        """
        x = x + self.attn(self.norm1(x))
        if self.use_engram:
            x = x + self.engram(idx, x)
        x = x + self.mlp(self.norm2(x))
        return x


class EngramGPT(nn.Module):
    """GPT variant with an Engram scaffold inserted into selected layers."""

    def __init__(
        self,
        vocab_size,
        embd_dims,
        n_head,
        n_layer,
        block_size,
        memory_size=65536,
        engram_layers=None,
        ngram=2,
        kernel_size=3,
        engram_heads=1,
    ):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)

        # Resolve the Engram placement schedule. Default mirrors the paper's
        # [1, n_layer // 2] pattern: one early layer (right after attention
        # has produced the first real context) and one near the middle. Layer
        # 0 is excluded because the pre-context hidden states carry no
        # meaning for the lookup to exploit.
        if engram_layers is None:
            engram_layers = {1, n_layer // 2}
        engram_layers = set(engram_layers)
        if 0 in engram_layers:
            raise ValueError("Engram cannot be placed on layer 0.")
        if any(i < 0 or i >= n_layer for i in engram_layers):
            raise ValueError(
                f"engram_layers entries must lie in [0, {n_layer}); got {engram_layers}."
            )
        self.engram_layers = engram_layers

        self.blocks = nn.ModuleList(
            [
                EngramBlock(
                    embd_dims,
                    n_head,
                    block_size,
                    memory_size=memory_size,
                    vocab_size=vocab_size,
                    use_engram=layer_idx in engram_layers,
                    ngram=ngram,
                    kernel_size=kernel_size,
                    engram_heads=engram_heads,
                )
                for layer_idx in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(embd_dims, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        """Run the GPT stack with optional Engram insertions.

        Args:
            idx: Token ids with shape (B, T) and dtype torch.long. These ids
                serve two roles in this scaffold:
                1. They index the standard token embedding table `wte`.
                2. They are passed into the Engram lookup path so memory
                   addresses can be derived from token identities or local
                   token context.
        Returns:
            Logits with shape (B, T, vocab_size).
        """
        x = self.wte(idx)
        x = self.norm0(x)
        for block in self.blocks:
            x = block(idx, x)
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
    print("Import GPT or EngramGPT from this module, or wire it into a run script.")
