"""
microgpt in PyTorch — GPU training on WikiText-103.
  DeepSeek Sparse Attention (DSA): lightning indexer selects top-k keys
  per query for sparse MLA. FWHT + FP8 STE on indexer activations for
  quantization-friendly scoring. RMSNorm, MLP with ReLU, Adam.

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

def fp8_ste(x):
    """Quantize to FP8 with straight-through estimator for gradients."""
    x_q = x.to(torch.float8_e4m3fn).to(x.dtype)
    return x + (x_q - x).detach()


def fwht(x):
    """Fast Walsh-Hadamard Transform along the last dimension (unnormalized, in-place).
    Last dimension must be a power of 2."""
    n = x.shape[-1]
    h = 1
    while h < n:
        xv = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        xv[..., 0, :].add_(xv[..., 1, :])           # a' = a + b
        xv[..., 1, :].mul_(-2).add_(xv[..., 0, :])  # b' = -2b + a' = a - b
        h *= 2
    return x

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

    def forward(self, x, topk_idx=None, return_attn=False):
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

        if topk_idx is None:
            # Full attention (used during warmup and generate prefill)
            attn = ((q @ k.transpose(-2, -1)) + (qr @ kr.transpose(-2, -1))) / math.sqrt(hd + rd)
            attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, T, D)
            out = self.wo(out)
            return (out, attn) if return_attn else out

        # Sparse attention: gather selected keys, values, rope keys
        K = topk_idx.shape[-1]
        flat_idx = topk_idx.reshape(B, -1)                                      # (B, T*K)
        idx_hd = flat_idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, hd)     # (B, H, T*K, hd)
        k_sel = k.gather(2, idx_hd).reshape(B, H, T, K, hd)
        v_sel = v.gather(2, idx_hd).reshape(B, H, T, K, hd)

        idx_rd = flat_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, rd)     # (B, 1, T*K, rd)
        kr_sel = kr.gather(2, idx_rd).reshape(B, 1, T, K, rd)

        scores = (torch.einsum('bhtd,bhtkd->bhtk', q, k_sel)
                + torch.einsum('bhtd,bhtkd->bhtk', qr, kr_sel.expand(-1, H, -1, -1, -1)))
        scores = scores / math.sqrt(hd + rd)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhtk,bhtkd->bhtd', attn, v_sel)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class LightningIndexer(nn.Module):
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
        q = self.wqup(ql).view(B, T, H, hd).transpose(1, 2)           # (B, H, T, hd)
        qr = apply_rotary_emb(self.wqr(ql).view(B, T, H, rd).transpose(1, 2), self.rope_cos, self.rope_sin)  # (B, H, T, rd)

        ckv = self.wkdown(x)
        k = self.wkup(ckv).view(B, T, 1, hd).transpose(1, 2)        # (B, 1, T, hd) — MQA single head
        kr = apply_rotary_emb(self.wkr(x).unsqueeze(1), self.rope_cos, self.rope_sin)  # (B, 1, T, rd)

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


class MLP(nn.Module):
    def __init__(self, embd_dims):
        super().__init__()
        self.fc1 = nn.Linear(embd_dims, 4 * embd_dims, bias=False)
        self.fc2 = nn.Linear(4 * embd_dims, embd_dims, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, embd_dims, n_head, latent_dims, rope_dims, block_size, top_k):
        super().__init__()
        self.top_k = top_k
        self.norm1 = RMSNorm(embd_dims)
        self.lindexer = LightningIndexer(embd_dims, n_head, latent_dims, rope_dims, block_size)
        self.attn = CausalAttention(embd_dims, n_head, latent_dims, rope_dims, block_size)
        self.norm2 = RMSNorm(embd_dims)
        self.mlp = MLP(embd_dims)

    def forward(self, x, warmup=False):
        lis = self.lindexer(x)
        nx = self.norm1(x)

        T = x.shape[1]
        if warmup:
            # Full attention, but detach output so only indexer gets gradients
            out, p = self.attn(nx, topk_idx=None, return_attn=True)
            x = x + out.detach()

            # Target: sum attention across heads, L1-normalize, detach
            p = p.detach().sum(dim=1)                           # (B, T, T)
            p = p / p.sum(dim=-1, keepdim=True)                 # L1 normalize

            # Re-mask indexer scores (ReLU zeros aren't -inf for softmax)
            lis = lis.masked_fill(self.attn.mask[:T, :T], float('-inf'))
            aux_loss = F.kl_div(F.log_softmax(lis, dim=-1), p, reduction='batchmean')
        else:
            _, topk_idx = torch.topk(lis, min(self.top_k, x.shape[1]), dim=-1)
            x = x + self.attn(nx, topk_idx)
            aux_loss = torch.tensor(0.0, device=x.device)

        x = x + self.mlp(self.norm2(x))
        return x, aux_loss


class GPT(nn.Module):
    def __init__(self, vocab_size, embd_dims, n_head, latent_dims, rope_dims, n_layer, block_size, top_k=64, warmup_steps=0):
        super().__init__()
        self.block_size = block_size
        self.warmup_steps = warmup_steps
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.wte = nn.Embedding(vocab_size, embd_dims)
        self.norm0 = RMSNorm(embd_dims)
        self.blocks = nn.ModuleList([
            Block(embd_dims, n_head, latent_dims, rope_dims, block_size, top_k) for _ in range(n_layer)
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
        warmup = self.training and self._step < self.warmup_steps
        x = self.wte(idx)
        x = self.norm0(x)
        total_aux = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, aux = block(x, warmup=warmup)
            total_aux = total_aux + aux
        if self.training:
            self._step += 1
        logits = self.lm_head(x)
        # During warmup: detach logits so CE doesn't train main model,
        # only KL aux_loss backprops through the indexer
        return (logits.detach(), total_aux) if warmup else logits

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens, temperature=0.8):
        """DSA inference: lightning indexer selects top-k keys per query,
        absorbed MLA projections applied only to selected entries."""
        B, T_start = start_ids.shape
        device = start_ids.device
        idx = start_ids

        # ── 1. Precompute absorbed matrices per layer ──
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

        # ── 2. Prefill: build caches for both indexer and attention ──
        kv_cache = []     # attention ckv: (B, T, latent_dims)
        kr_cache = []     # attention kr:  (B, 1, T, rd)
        li_k_cache = []   # indexer post-FWHT k:  (B, 1, T, li_hd)
        li_kr_cache = []  # indexer post-FWHT kr: (B, 1, T, li_rd)

        T = T_start
        x = self.wte(idx)
        x = self.norm0(x)

        for i, block in enumerate(self.blocks):
            a = block.attn
            li = block.lindexer
            hd = a.head_dim
            rd = a.rope_dim
            H = a.n_head

            # --- Lightning indexer ---
            li_ql = li.wqdown(x)
            li_q = li.wqup(li_ql).view(B, T, H, li.head_dim).transpose(1, 2)
            li_qr = apply_rotary_emb(
                li.wqr(li_ql).view(B, T, H, li.rope_dim).transpose(1, 2),
                li.rope_cos, li.rope_sin)

            li_ck = li.wkdown(x)
            li_k = li.wkup(li_ck).view(B, T, 1, li.head_dim).transpose(1, 2)
            li_kr = apply_rotary_emb(li.wkr(x).unsqueeze(1), li.rope_cos, li.rope_sin)

            li_q = fp8_ste(fwht(li_q.contiguous()) / math.sqrt(li.head_dim))
            li_k = fp8_ste(fwht(li_k.contiguous()) / math.sqrt(li.head_dim))
            li_qr = fp8_ste(fwht(li_qr.contiguous()) / math.sqrt(li.rope_dim))
            li_kr = fp8_ste(fwht(li_kr.contiguous()) / math.sqrt(li.rope_dim))

            li_k_cache.append(li_k)
            li_kr_cache.append(li_kr)

            lis = (li_q @ li_k.transpose(-2, -1)) + (li_qr @ li_kr.transpose(-2, -1))
            lis = lis.masked_fill(li.mask[:T, :T], float('-inf'))
            lis = F.relu(lis)
            hw = F.sigmoid(li.wh(li_ql)).transpose(1, 2).unsqueeze(-1)
            lis = (hw * lis).sum(dim=1)                                     # (B, T, T)

            K = min(block.top_k, T)
            _, topk_idx = torch.topk(lis, K, dim=-1)                        # (B, T, K)

            # --- Sparse attention with absorbed matrices ---
            nx = block.norm1(x)
            ql = a.wqdown(nx)                                              # (B, T, latent)
            qr = apply_rotary_emb(
                a.wqr(ql).view(B, T, H, rd).transpose(1, 2),
                a.rope_cos, a.rope_sin)                                    # (B, H, T, rd)
            ckv = a.wkvdown(nx)                                            # (B, T, latent)
            kr = apply_rotary_emb(a.wkr(nx).unsqueeze(1), a.rope_cos, a.rope_sin)  # (B, 1, T, rd)

            kv_cache.append(ckv)
            kr_cache.append(kr)

            # Gather selected ckv and kr
            flat_idx = topk_idx.reshape(B, -1)                             # (B, T*K)
            idx_ckv = flat_idx.unsqueeze(-1).expand(-1, -1, ckv.shape[-1]) # (B, T*K, latent)
            ckv_sel = ckv.gather(1, idx_ckv).reshape(B, T, K, -1)         # (B, T, K, latent)

            idx_kr = flat_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, rd)
            kr_sel = kr.gather(2, idx_kr).reshape(B, 1, T, K, rd)         # (B, 1, T, K, rd)

            # Absorbed scores: ql @ W_qk @ ckv_sel + qr @ kr_sel
            q_abs = ql.unsqueeze(1) @ absorbed_qk[i]                      # (B, H, T, latent)
            content = torch.einsum('bhtl,btkl->bhtk', q_abs, ckv_sel)
            rope = torch.einsum('bhtd,bhtkd->bhtk', qr, kr_sel.expand(-1, H, -1, -1, -1))
            scores = (content + rope) / math.sqrt(hd + rd)
            attn = F.softmax(scores, dim=-1)                               # (B, H, T, K)

            # Output: attn @ ckv_sel through absorbed W_vo
            weighted = torch.einsum('bhtk,btkl->bhtl', attn, ckv_sel)     # (B, H, T, latent)
            out = (weighted @ absorbed_vo[i]).sum(dim=1)                   # (B, T, embd)

            x = x + out
            x = x + block.mlp(block.norm2(x))

        # ── 3. Decode: one token at a time ──
        for step in range(max_new_tokens):
            pos = T_start + step
            x_new = self.wte(idx[:, -1:])
            x_new = self.norm0(x_new)

            for i, block in enumerate(self.blocks):
                a = block.attn
                li = block.lindexer
                hd = a.head_dim
                rd = a.rope_dim
                H = a.n_head
                cos_pos = a.rope_cos[pos:pos+1]
                sin_pos = a.rope_sin[pos:pos+1]
                li_cos = li.rope_cos[pos:pos+1]
                li_sin = li.rope_sin[pos:pos+1]

                # --- Indexer: extend cache, score new query ---
                li_ql = li.wqdown(x_new)
                li_q = li.wqup(li_ql).view(B, 1, H, li.head_dim).transpose(1, 2)
                li_qr = apply_rotary_emb(
                    li.wqr(li_ql).view(B, 1, H, li.rope_dim).transpose(1, 2),
                    li_cos, li_sin)

                li_ck = li.wkdown(x_new)
                li_k_new = li.wkup(li_ck).view(B, 1, 1, li.head_dim).transpose(1, 2)
                li_kr_new = apply_rotary_emb(li.wkr(x_new).unsqueeze(1), li_cos, li_sin)

                li_q = fp8_ste(fwht(li_q.contiguous()) / math.sqrt(li.head_dim))
                li_k_new = fp8_ste(fwht(li_k_new.contiguous()) / math.sqrt(li.head_dim))
                li_qr = fp8_ste(fwht(li_qr.contiguous()) / math.sqrt(li.rope_dim))
                li_kr_new = fp8_ste(fwht(li_kr_new.contiguous()) / math.sqrt(li.rope_dim))

                li_k_cache[i] = torch.cat([li_k_cache[i], li_k_new], dim=2)
                li_kr_cache[i] = torch.cat([li_kr_cache[i], li_kr_new], dim=2)

                # (B, H, 1, T_cur) — no causal mask needed for single query
                lis = (li_q @ li_k_cache[i].transpose(-2, -1)
                     + li_qr @ li_kr_cache[i].transpose(-2, -1))
                lis = F.relu(lis)
                hw = F.sigmoid(li.wh(li_ql)).transpose(1, 2).unsqueeze(-1)  # (B, H, 1, 1)
                lis = (hw * lis).sum(dim=1)                                  # (B, 1, T_cur)

                T_cur = lis.shape[-1]
                K = min(block.top_k, T_cur)
                _, topk_idx = torch.topk(lis, K, dim=-1)                     # (B, 1, K)

                # --- Sparse attention with absorbed matrices ---
                nx = block.norm1(x_new)
                ql = a.wqdown(nx)                                          # (B, 1, latent)
                qr = apply_rotary_emb(
                    a.wqr(ql).view(B, 1, H, rd).transpose(1, 2),
                    cos_pos, sin_pos)                                      # (B, H, 1, rd)

                ckv = a.wkvdown(nx)                                        # (B, 1, latent)
                kv_cache[i] = torch.cat([kv_cache[i], ckv], dim=1)
                ckv_all = kv_cache[i]
                kr = apply_rotary_emb(a.wkr(nx).unsqueeze(1), cos_pos, sin_pos)  # (B, 1, 1, rd)
                kr_cache[i] = torch.cat([kr_cache[i], kr], dim=2)
                kr_all = kr_cache[i]

                # Gather selected entries
                flat_idx = topk_idx.reshape(B, -1)                         # (B, K)
                idx_ckv = flat_idx.unsqueeze(-1).expand(-1, -1, ckv_all.shape[-1])
                ckv_sel = ckv_all.gather(1, idx_ckv).reshape(B, 1, K, -1) # (B, 1, K, latent)

                idx_kr = flat_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, rd)
                kr_sel = kr_all.gather(2, idx_kr).reshape(B, 1, 1, K, rd) # (B, 1, 1, K, rd)

                q_abs = ql.unsqueeze(1) @ absorbed_qk[i]                  # (B, H, 1, latent)
                content = torch.einsum('bhtl,btkl->bhtk', q_abs, ckv_sel)
                rope = torch.einsum('bhtd,bhtkd->bhtk', qr, kr_sel.expand(-1, H, -1, -1, -1))
                scores = (content + rope) / math.sqrt(hd + rd)
                attn = F.softmax(scores, dim=-1)

                weighted = torch.einsum('bhtk,btkl->bhtl', attn, ckv_sel)
                out = (weighted @ absorbed_vo[i]).sum(dim=1)

                x_new = x_new + out
                x_new = x_new + block.mlp(block.norm2(x_new))

            logits = self.lm_head(x_new)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=-1)

        return idx


if __name__ == '__main__':
    print("Use run_attention_comparison.py to train and evaluate.")