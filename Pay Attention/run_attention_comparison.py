"""
Runner script for comparing attention variants (MHA, MQA, GQA, MLA).
Trains each selected variant, then plots a 2x2 comparison:
  loss curves | KV cache memory
  param count | inference time

Usage:
  # Sequential (all on one GPU):
  python run_attention_comparison.py

  # Parallel (one method per GPU, then combine):
  CUDA_VISIBLE_DEVICES=0 python run_attention_comparison.py --methods mha &
  CUDA_VISIBLE_DEVICES=1 python run_attention_comparison.py --methods mqa &
  CUDA_VISIBLE_DEVICES=2 python run_attention_comparison.py --methods gqa &
  CUDA_VISIBLE_DEVICES=3 python run_attention_comparison.py --methods mla &
  wait
  python run_attention_comparison.py --plot_only
"""

import os
import math
import time
import argparse
import importlib.util
import yaml

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODULE_FILES = {
    'mha': 'microgpt-mha.py',
    'mqa': 'microgpt-mqa.py',
    'gqa': 'microgpt-gqa.py',
    'mla': 'microgpt-mla.py',
}

LABELS = {'mha': 'MHA', 'mqa': 'MQA', 'gqa': 'GQA', 'mla': 'MLA'}
COLORS = {'mha': '#1f77b4', 'mqa': '#ff7f0e', 'gqa': '#2ca02c', 'mla': '#d62728'}


# --- Helpers ---

def load_module(method):
    filepath = os.path.join(SCRIPT_DIR, MODULE_FILES[method])
    spec = importlib.util.spec_from_file_location(f'microgpt_{method}', filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(method, module, args, vocab_size):
    if method in ('mha', 'mqa'):
        return module.GPT(vocab_size, args.embd_dims, args.n_head,
                          args.n_layer, args.block_size)
    elif method == 'gqa':
        return module.GPT(vocab_size, args.embd_dims, args.n_qhead, args.n_kvhead,
                          args.n_layer, args.block_size)
    elif method == 'mla':
        return module.GPT(vocab_size, args.embd_dims, args.n_head,
                          args.latent_dims, args.rope_dims,
                          args.n_layer, args.block_size)


def compute_kv_cache_bytes(method, args):
    B = args.infer_batch
    T = 1 + args.max_new_tokens
    L = args.n_layer
    if method == 'mha':
        hd = args.embd_dims // args.n_head
        return 2 * L * B * args.n_head * T * hd * 4
    elif method == 'mqa':
        hd = args.embd_dims // args.n_head
        return 2 * L * B * 1 * T * hd * 4
    elif method == 'gqa':
        hd = args.embd_dims // args.n_qhead
        return 2 * L * B * args.n_kvhead * T * hd * 4
    elif method == 'mla':
        return L * B * T * (args.latent_dims + args.rope_dims) * 4


def train_model(model, train_loader, val_loader, vocab_size, args, device, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    loss_history = []
    train_iter = iter(train_loader)
    t0 = time.time()

    model.train()
    for step in range(args.num_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 100 == 0 or step == 0:
            loss_history.append(loss.item())
            elapsed = time.time() - t0
            tps = (step + 1) * args.batch_size * args.block_size / elapsed
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{label}] step {step+1:5d}/{args.num_steps} | "
                  f"loss {loss.item():.4f} | {tps:,.0f} tok/s | lr {lr_now:.2e}")

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            val_losses.append(loss.item())
            if len(val_losses) >= 50:
                break
    val_loss = sum(val_losses) / len(val_losses)

    return loss_history, val_loss


# --- DataLoader (shared across all variants) ---

class TextDataset(torch.utils.data.Dataset):
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


# --- Main ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare attention variants')
    parser.add_argument('--config', type=str, default=os.path.join(SCRIPT_DIR, 'config.yaml'),
                        help='Path to YAML config file')
    parser.add_argument('--methods', nargs='+', choices=['mha', 'mqa', 'gqa', 'mla'],
                        help='Override methods from config')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory for saved results (default: script dir)')
    parser.add_argument('--plot_only', action='store_true',
                        help='Skip training, load saved results and plot')
    cli_args = parser.parse_args()

    # Load config from YAML, then apply CLI overrides
    with open(cli_args.config) as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)
    if cli_args.methods:
        args.methods = cli_args.methods
    args.save_dir = cli_args.save_dir
    args.plot_only = cli_args.plot_only

    save_dir = args.save_dir or SCRIPT_DIR
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.plot_only:
        print(f"device: {device}")
        print(f"methods: {', '.join(LABELS[m] for m in args.methods)}")

        # Load dataset once
        print("Loading WikiText-103...")
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
        enc = tiktoken.get_encoding('gpt2')
        vocab_size = enc.n_vocab

        print("Tokenizing...")
        train_tokens = torch.tensor(enc.encode('\n'.join(ds['train']['text'])), dtype=torch.long)
        val_tokens = torch.tensor(enc.encode('\n'.join(ds['validation']['text'])), dtype=torch.long)
        print(f"train: {len(train_tokens):,} tokens, val: {len(val_tokens):,} tokens")

        train_ds = TextDataset(train_tokens, args.block_size)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_ds = TextDataset(val_tokens, args.block_size)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, drop_last=True)

        # Train each variant
        for method in args.methods:
            label = LABELS[method]
            print(f"\n{'='*60}")
            print(f"  Training: {label}")
            print(f"{'='*60}")

            module = load_module(method)
            model = build_model(method, module, args, vocab_size).to(device)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  params: {num_params:,}")

            loss_history, val_loss = train_model(
                model, train_loader, val_loader, vocab_size, args, device, label)

            kv_bytes = compute_kv_cache_bytes(method, args)

            # Inference timing (average over multiple runs)
            model.eval()
            start = torch.tensor([[enc.encode('\n')[0]]] * args.infer_batch, device=device)
            n_runs = 5
            # Warmup
            with torch.no_grad():
                model.generate(start, max_new_tokens=args.max_new_tokens, temperature=0.8)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            infer_times = []
            for _ in range(n_runs):
                t_infer = time.time()
                with torch.no_grad():
                    model.generate(start, max_new_tokens=args.max_new_tokens, temperature=0.8)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                infer_times.append(time.time() - t_infer)
            infer_time = sum(infer_times) / n_runs

            result = {
                'loss_history': loss_history,
                'val_loss': val_loss,
                'infer_time': infer_time,
                'kv_cache_bytes': kv_bytes,
                'num_params': num_params,
            }

            # Save result metrics
            result_path = os.path.join(save_dir, f'results_{method}.pt')
            torch.save(result, result_path)

            # Save model weights
            model_dir = os.path.join(SCRIPT_DIR, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'{method}.pt')
            torch.save(model.state_dict(), model_path)

            print(f"  val loss: {val_loss:.4f} (ppl {math.exp(val_loss):.1f})")
            print(f"  inference: {infer_time:.3f}s (B={args.infer_batch}, T={args.max_new_tokens})")
            print(f"  KV cache: {kv_bytes / 1024 / 1024:.2f} MB")
            print(f"  model saved to {model_path}")

            del model, module
            torch.cuda.empty_cache()

    # --- Load results and plot ---
    results = {}
    for method in args.methods:
        path = os.path.join(save_dir, f'results_{method}.pt')
        if os.path.exists(path):
            results[method] = torch.load(path, weights_only=False)
        else:
            print(f"Warning: no results for {LABELS[method]} at {path}")

    if not results:
        print("No results to plot.")
        exit(1)

    methods_run = list(results.keys())
    labels = [LABELS[m] for m in methods_run]
    colors = [COLORS[m] for m in methods_run]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: loss curves
    ax = axes[0, 0]
    for m, c in zip(methods_run, colors):
        h = results[m]['loss_history']
        # First entry is step 1, then every 100 steps
        steps_x = np.array([1] + list(range(100, 100 * len(h), 100)))[:len(h)]
        ax.plot(steps_x, h, label=LABELS[m], color=c)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: KV cache memory
    ax = axes[0, 1]
    kv_mb = [results[m]['kv_cache_bytes'] / 1024 / 1024 for m in methods_run]
    bars = ax.bar(labels, kv_mb, color=colors)
    ax.set_ylabel('MB')
    ax.set_title(f'KV Cache (B={args.infer_batch}, T={1+args.max_new_tokens})')
    ax.bar_label(bars, fmt='%.2f')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: parameter count
    ax = axes[1, 0]
    params_m = [results[m]['num_params'] / 1e6 for m in methods_run]
    bars = ax.bar(labels, params_m, color=colors)
    ax.set_ylabel('Millions')
    ax.set_title('Parameter Count')
    ax.bar_label(bars, fmt='%.2f')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: inference time
    ax = axes[1, 1]
    times = [results[m]['infer_time'] for m in methods_run]
    bars = ax.bar(labels, times, color=colors)
    ax.set_ylabel('Seconds')
    ax.set_title(f'Inference Time (B={args.infer_batch}, T={args.max_new_tokens})')
    ax.bar_label(bars, fmt='%.3f')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Attention Comparison | {args.n_layer}L {args.embd_dims}D '
                 f'{args.block_size}T {args.num_steps} steps',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'attention_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
