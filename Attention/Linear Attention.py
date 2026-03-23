"""
microgpt in PyTorch — GPU training on WikiText-103.
  RMSNorm, chunkwise causal linear attention, MLP with ReLU, Adam.
  Supports multi-GPU via distributed chunk prefix scan.

Requirements: pip install torch datasets tiktoken matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# --- Distributed helpers (gradient-aware) ---

class _GatherChunks(torch.autograd.Function):
    """all_gather along the chunk dim (dim=1), with gradient split on backward."""
    @staticmethod
    def forward(ctx, x):
        world = dist.get_world_size()
        out = [torch.zeros_like(x) for _ in range(world)]
        dist.all_gather(out, x.contiguous())
        ctx.world = world
        return torch.cat(out, dim=1)

    @staticmethod
    def backward(ctx, grad):
        rank = dist.get_rank()
        return grad.chunk(ctx.world, dim=1)[rank].contiguous()


class _PrefixExchange(torch.autograd.Function):
    """
    Exclusive cross-GPU prefix sum of per-GPU KV state totals.

    Forward:  prefix[rank] = sum(total[0], ..., total[rank-1])
    Backward: grad_total[rank] = sum(grad_prefix[rank+1], ..., grad_prefix[W-1])
    """
    @staticmethod
    def forward(ctx, local_total):
        world = dist.get_world_size()
        rank = dist.get_rank()
        ctx.world, ctx.rank = world, rank

        totals = [torch.zeros_like(local_total) for _ in range(world)]
        dist.all_gather(totals, local_total.contiguous())

        prefix = torch.zeros_like(local_total)
        for r in range(rank):
            prefix = prefix + totals[r]
        return prefix

    @staticmethod
    def backward(ctx, grad_prefix):
        world, rank = ctx.world, ctx.rank
        grads = [torch.zeros_like(grad_prefix) for _ in range(world)]
        dist.all_gather(grads, grad_prefix.contiguous())

        grad_total = torch.zeros_like(grad_prefix)
        for r in range(rank + 1, world):
            grad_total = grad_total + grads[r]
        return grad_total


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


# --- Model ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalAttention(nn.Module):
    def __init__(self, embd_dims, num_chunks):
        super().__init__()
        self.wq = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wk = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wv = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wo = nn.Linear(embd_dims, embd_dims, bias=False)
        self.C = num_chunks

    def forward(self, x):
        B, T, D = x.shape
        C = self.C
        L = T // C

        q = self.wq(x).view(B, C, L, D)
        k = self.wk(x).view(B, C, L, D)
        v = self.wv(x).view(B, C, L, D)

        use_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        if use_dist:
            out = self._forward_distributed(q, k, v, B, C, L, D)
        else:
            out = self._forward_local(q, k, v, B, C, L, D)

        return self.wo(out.reshape(B, T, D))

    def _forward_local(self, q, k, v, B, C, L, D):
        """Single-GPU chunkwise causal linear attention."""
        # Intra-chunk: causal via cumsum within each chunk
        kv = torch.einsum('bcti,bctj->bctij', k, v)       # (B, C, L, D, D)
        kv_cumsum = torch.cumsum(kv, dim=2)                 # prefix sum along time-in-chunk
        out_intra = torch.einsum('bcti,bctij->bctj', q, kv_cumsum)

        # Inter-chunk: exclusive prefix scan of per-chunk KV sums
        chunk_kv = kv.sum(dim=2)                            # (B, C, D, D)
        S = torch.zeros_like(chunk_kv)
        S[:, 1:] = torch.cumsum(chunk_kv[:, :-1], dim=1)   # S[c] = sum(chunk_kv[0..c-1])
        out_inter = torch.einsum('bcti,bcij->bctj', q, S)

        return out_intra + out_inter

    def _forward_distributed(self, q, k, v, B, C, L, D):
        """
        Multi-GPU: each GPU computes C_local = C // world_size chunks.

        Communication (2 collective ops per layer):
          1. all_gather for cross-GPU prefix scan  (D*D floats per GPU)
          2. all_gather to reassemble chunk outputs (C_local*L*D floats per GPU)
        """
        rank = dist.get_rank()
        world = dist.get_world_size()
        C_local = C // world
        c0 = rank * C_local

        # Slice this GPU's chunks
        q_l = q[:, c0:c0+C_local].contiguous()
        k_l = k[:, c0:c0+C_local].contiguous()
        v_l = v[:, c0:c0+C_local].contiguous()

        # ---- Intra-chunk (fully local, no communication) ----
        kv_l = torch.einsum('bcti,bctj->bctij', k_l, v_l)  # (B, C_local, L, D, D)
        kv_cumsum_l = torch.cumsum(kv_l, dim=2)
        out_intra = torch.einsum('bcti,bctij->bctj', q_l, kv_cumsum_l)

        # ---- Inter-chunk ----
        chunk_kv_l = kv_l.sum(dim=2)                        # (B, C_local, D, D)

        # Local exclusive prefix scan
        S_l = torch.zeros_like(chunk_kv_l)
        if C_local > 1:
            S_l[:, 1:] = torch.cumsum(chunk_kv_l[:, :-1], dim=1)

        # Cross-GPU prefix: sum of KV states from all earlier GPUs
        local_total = chunk_kv_l.sum(dim=1)                 # (B, D, D)
        gpu_prefix = _PrefixExchange.apply(local_total)     # (B, D, D)
        S_l = S_l + gpu_prefix.unsqueeze(1)                 # broadcast to all local chunks

        out_inter = torch.einsum('bcti,bcij->bctj', q_l, S_l)

        out_local = out_intra + out_inter                   # (B, C_local, L, D)

        # ---- Reassemble full sequence from all GPUs ----
        return _GatherChunks.apply(out_local)               # (B, C, L, D)


class DeltaAttention(nn.Module):
    """
    Delta Update Rule attention (chunkwise parallel form).

    Update:  S_t = A_t S_{t-1} + B_t
    where    A_t = I - beta_t * k_t^T k_t
             B_t = beta_t * k_t^T v_t
    Output:  o_t = q_t S_t
    """
    def __init__(self, embd_dims, num_chunks):
        super().__init__()
        self.wq = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wk = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wv = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wo = nn.Linear(embd_dims, embd_dims, bias=False)
        self.w_beta = nn.Linear(embd_dims, embd_dims, bias=False)  # learns beta per-dim
        self.C = num_chunks

    def _compute_beta(self, x):
        """beta = sigmoid(x W_beta), shape (B, T, D) -> (B, T, D)."""
        return torch.sigmoid(self.w_beta(x))

    def _forward_recurrent(self, q, k, v, beta):
        """
        Naive sequential recurrence (reference impl, O(T d^2)).
        Useful for correctness checking against the chunkwise version.
        """
        B, T, D = q.shape
        S = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
        out = torch.zeros_like(q)
        for t in range(T):
            # A_t S_{t-1} + B_t
            # = (I - beta_t k_t^T k_t) S_{t-1} + beta_t k_t^T v_t
            # = S_{t-1} + beta_t k_t^T (v_t - k_t S_{t-1})
            k_t = k[:, t]                                  # (B, D)
            v_t = v[:, t]                                   # (B, D)
            b_t = beta[:, t]                                # (B, D)

            v_old = torch.einsum('bi,bij->bj', k_t, S)     # k_t S_{t-1}
            v_new = (1 - b_t) * v_old + b_t * v_t          # interpolate
            # S_t = S_{t-1} + k_t^T (v_new - v_old)
            S = S + torch.einsum('bi,bj->bij', k_t, v_new - v_old)

            out[:, t] = torch.einsum('bi,bij->bj', q[:, t], S)
        return out

    def _intra_chunk(self, q, k, v, beta):
        """
        Within-chunk parallel computation using the scan S_t = A_t S_{t-1} + B_t.
        Returns per-chunk outputs and final chunk states.

        q, k, v, beta: (B, C, L, D)
        Returns: out (B, C, L, D), S_final (B, C, D, D)
        """
        B, C, L, D = q.shape

        # Precompute A_t and B_t for all timesteps
        # A_t = I - beta_t * k_t^T k_t   (B, C, L, D, D)
        # B_t = beta_t * k_t^T v_t        (B, C, L, D, D)
        bk = beta * k                                              # (B, C, L, D)
        A = -torch.einsum('bcli,bclj->bclij', bk, k)              # -beta k^T k
        # Add identity along diagonal
        eye = torch.eye(D, device=q.device, dtype=q.dtype)
        A = A + eye.reshape(1, 1, 1, D, D)                        # I - beta k^T k
        B_mat = torch.einsum('bcli,bclj->bclij', bk, v)           # beta k^T v

        # Sequential scan within each chunk (parallel across chunks)
        S = torch.zeros(B, C, D, D, device=q.device, dtype=q.dtype)
        out = torch.zeros_like(q)
        A_prod = torch.eye(D, D).expand(B, C, D, D)
        for t in range(L):
            # S = A_t @ S + B_t
            S = torch.einsum('bcij,bcjk->bcik', A[:, :, t], S) + B_mat[:, :, t]
            out[:, :, t] = torch.einsum('bci,bcij->bcj', q[:, :, t], S)
            A_prod = A[:, :, t] @ A_prod

        S_final = S  # (B, C, D, D) — final state per chunk
        return out, S_final, A_prod

    def _inter_chunk(self, q, k, beta, S_finals, A_prod):
        """
        Cross-chunk prefix scan: propagate chunk-end states to subsequent chunks.

        S_finals: (B, C, D, D) — final state of each chunk.
        A_prod:   (B, C, D, D) — product of all A matrices within each chunk.

        Returns: out_inter (B, C, L, D) — inter-chunk correction.
        """
        B, C, L, D = q.shape

        # Exclusive prefix scan with A-product propagation
        S_prefix = torch.zeros(B, C, D, D, device=q.device, dtype=q.dtype)
        for c in range(1, C):
            S_prefix[:, c] = A_prod[:, c] @ S_prefix[:, c - 1] + S_finals[:, c - 1]

        # q @ S_prefix for each token
        out_inter = torch.einsum('bcti,bcij->bctj', q, S_prefix)
        return out_inter

    def forward(self, x):
        B, T, D = x.shape
        C = self.C
        L = T // C

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        beta = self._compute_beta(x)                          # (B, T, D)

        # Reshape into chunks
        q = q.view(B, C, L, D)
        k = k.view(B, C, L, D)
        v = v.view(B, C, L, D)
        beta = beta.view(B, C, L, D)

        # Intra-chunk: scan within each chunk
        out_intra, S_finals, A_prod = self._intra_chunk(q, k, v, beta)

        # Inter-chunk: prefix scan across chunks
        out_inter = self._inter_chunk(q, k, beta, S_finals, A_prod)

        out = out_intra + out_inter
        return self.wo(out.reshape(B, T, D))


class WKVAttention(nn.Module):
    """
    RWKV-style WKV linear attention (chunkwise parallel form).

    Update:  S_t = diag(w_t) * S_{t-1} + k_t^T v_t
    Output:  o_t = q_t S_t

    w_t is a per-dimension learned decay (element-wise, not a full matrix).
    """
    def __init__(self, embd_dims, num_chunks):
        super().__init__()
        self.wq = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wk = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wv = nn.Linear(embd_dims, embd_dims, bias=False)
        self.wo = nn.Linear(embd_dims, embd_dims, bias=False)
        self.w_decay = nn.Linear(embd_dims, embd_dims, bias=False)  # learns per-dim decay
        self.C = num_chunks

    def _compute_decay(self, x):
        """w = sigmoid(x W_decay), shape (B, T, D) -> (B, T, D). Values in (0,1)."""
        return torch.sigmoid(self.w_decay(x))

    def _forward_recurrent(self, q, k, v, w):
        """Naive sequential recurrence (reference impl)."""
        B, T, D = q.shape
        S = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
        out = torch.zeros_like(q)
        for t in range(T):
            w_t = w[:, t]                                      # (B, D)
            # S_t = diag(w_t) * S_{t-1} + k_t^T v_t
            S = w_t.unsqueeze(-1) * S + torch.einsum('bi,bj->bij', k[:, t], v[:, t])
            out[:, t] = torch.einsum('bi,bij->bj', q[:, t], S)
        return out

    def _intra_chunk(self, q, k, v, w):
        """
        Within-chunk computation with decay.
        q, k, v, w: (B, C, L, D)
        Returns: out (B, C, L, D), S_final (B, C, D, D), W_prod (B, C, D)
        """
        B, C, L, D = q.shape

        S = torch.zeros(B, C, D, D, device=q.device, dtype=q.dtype)
        out = torch.zeros_like(q)
        W_prod = torch.ones(B, C, D, device=q.device, dtype=q.dtype)

        for t in range(L):
            w_t = w[:, :, t]                                   # (B, C, D)
            # S = diag(w_t) * S + k_t^T v_t
            S = w_t.unsqueeze(-1) * S + torch.einsum('bci,bcj->bcij', k[:, :, t], v[:, :, t])
            out[:, :, t] = torch.einsum('bci,bcij->bcj', q[:, :, t], S)
            W_prod = W_prod * w_t                              # cumulative decay product

        return out, S, W_prod

    def _inter_chunk(self, q, w, S_finals, W_prod):
        """
        Cross-chunk prefix scan with decay propagation.
        W_prod: (B, C, D) — product of all decays within each chunk.
        """
        B, C, L, D = q.shape

        # Exclusive prefix scan: S_prefix[c] = decayed sum of all prior chunk states
        S_prefix = torch.zeros(B, C, D, D, device=q.device, dtype=q.dtype)
        for c in range(1, C):
            # Apply this chunk's total decay to previous prefix, then add previous chunk's state
            S_prefix[:, c] = W_prod[:, c].unsqueeze(-1) * S_prefix[:, c - 1] + S_finals[:, c - 1]

        out_inter = torch.einsum('bcti,bcij->bctj', q, S_prefix)
        return out_inter

    def forward(self, x):
        B, T, D = x.shape
        C = self.C
        L = T // C

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        w = self._compute_decay(x)                            # (B, T, D)

        q = q.view(B, C, L, D)
        k = k.view(B, C, L, D)
        v = v.view(B, C, L, D)
        w = w.view(B, C, L, D)

        out_intra, S_finals, W_prod = self._intra_chunk(q, k, v, w)
        out_inter = self._inter_chunk(q, w, S_finals, W_prod)

        out = out_intra + out_inter
        return self.wo(out.reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, embd_dims):
        super().__init__()
        self.fc1 = nn.Linear(embd_dims, 4 * embd_dims, bias=False)
        self.fc2 = nn.Linear(4 * embd_dims, embd_dims, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, embd_dims, num_chunks):
        super().__init__()
        self.norm1 = RMSNorm(embd_dims)
        self.attn = CausalAttention(embd_dims, num_chunks)
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embd_dims, n_layer, block_size, num_chunks=4):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)
        self.blocks = nn.ModuleList([
            Block(embd_dims, num_chunks) for _ in range(n_layer)
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
