"""
Compare baseline GPT vs Attention Residuals GPT on WikiText-103.
Plots: loss curves, parameter count, inference time, gradient norm over training.

Usage:
  python run_attention_residuals_comparison.py
  python run_attention_residuals_comparison.py --plot_only
"""

import math
import os
import time
import argparse

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

# --- Config ---
DEFAULTS = dict(
    embd_dims=128,
    n_head=4,
    n_layer=8,
    block_size=128,
    batch_size=32,
    num_steps=2000,
    lr=3e-4,
    infer_batch=4,
    max_new_tokens=64,
)

MODELS = {
    'gpt': {'label': 'Baseline GPT', 'color': '#1f77b4'},
    'argpt': {'label': 'Attention Residuals GPT', 'color': '#d62728'},
}


def load_data(block_size, batch_size):
    print("Loading WikiText-103...")
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
    enc = tiktoken.get_encoding('gpt2')
    vocab_size = enc.n_vocab

    print("Tokenizing...")
    train_tokens = torch.tensor(
        enc.encode('\n'.join(ds['train']['text'])), dtype=torch.long)
    val_tokens = torch.tensor(
        enc.encode('\n'.join(ds['validation']['text'])), dtype=torch.long)
    print(f"train: {len(train_tokens):,} tokens, val: {len(val_tokens):,} tokens")

    from importlib import import_module
    spec = __import__('importlib').util.spec_from_file_location(
        'ar', os.path.join(SCRIPT_DIR, 'Attention Residuals.py'))
    mod = __import__('importlib').util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    train_ds = mod.TextDataset(train_tokens, block_size)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_ds = mod.TextDataset(val_tokens, block_size)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, vocab_size, enc, mod


def train_model(model, train_loader, val_loader, vocab_size, args, device, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    loss_history = []
    grad_norms = []
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

        # Track gradient norm before clipping
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(total_norm.item())

        optimizer.step()
        scheduler.step()

        if (step + 1) % 100 == 0 or step == 0:
            loss_history.append(loss.item())
            elapsed = time.time() - t0
            tps = (step + 1) * args.batch_size * args.block_size / elapsed
            print(f"  [{label}] step {step+1:5d}/{args.num_steps} | "
                  f"loss {loss.item():.4f} | grad {total_norm:.4f} | {tps:,.0f} tok/s")

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

    return loss_history, val_loss, grad_norms


def measure_inference(model, enc, device, args):
    model.eval()
    start = torch.tensor(
        [[enc.encode('\n')[0]]] * args.infer_batch, device=device)

    # Warmup
    with torch.no_grad():
        model.generate(start, max_new_tokens=args.max_new_tokens, temperature=0.8)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    n_runs = 5
    infer_times = []
    for _ in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            model.generate(start, max_new_tokens=args.max_new_tokens, temperature=0.8)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        infer_times.append(time.time() - t0)

    return sum(infer_times) / n_runs


def plot_results(results, args):
    methods = list(results.keys())
    labels = [MODELS[m]['label'] for m in methods]
    colors = [MODELS[m]['color'] for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: loss curves
    ax = axes[0, 0]
    for m, c in zip(methods, colors):
        h = results[m]['loss_history']
        steps_x = np.array([1] + list(range(100, 100 * len(h), 100)))[:len(h)]
        ax.plot(steps_x, h, label=MODELS[m]['label'], color=c)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: gradient norms (smoothed)
    ax = axes[0, 1]
    window = 50
    for m, c in zip(methods, colors):
        g = results[m]['grad_norms']
        smoothed = np.convolve(g, np.ones(window) / window, mode='valid')
        ax.plot(smoothed, label=MODELS[m]['label'], color=c, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: parameter count
    ax = axes[1, 0]
    params_m = [results[m]['num_params'] / 1e6 for m in methods]
    bars = ax.bar(labels, params_m, color=colors)
    ax.set_ylabel('Millions')
    ax.set_title('Parameter Count')
    ax.bar_label(bars, fmt='%.2f')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: inference time
    ax = axes[1, 1]
    times = [results[m]['infer_time'] for m in methods]
    bars = ax.bar(labels, times, color=colors)
    ax.set_ylabel('Seconds')
    ax.set_title(f'Inference Time (B={args.infer_batch}, T={args.max_new_tokens})')
    ax.bar_label(bars, fmt='%.3f')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Baseline GPT vs Attention Residuals | {args.n_layer}L {args.embd_dims}D '
                 f'{args.block_size}T {args.num_steps} steps',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'attention_residuals_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare GPT vs Attention Residuals GPT')
    parser.add_argument('--plot_only', action='store_true')
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    cli_args = parser.parse_args()

    args = argparse.Namespace(**DEFAULTS)
    if cli_args.num_steps:
        args.num_steps = cli_args.num_steps
    if cli_args.n_layer:
        args.n_layer = cli_args.n_layer

    save_dir = SCRIPT_DIR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    if not cli_args.plot_only:
        train_loader, val_loader, vocab_size, enc, mod = load_data(
            args.block_size, args.batch_size)

        for method in ['gpt', 'argpt']:
            label = MODELS[method]['label']
            print(f"\n{'='*60}")
            print(f"  Training: {label}")
            print(f"{'='*60}")

            if method == 'gpt':
                model = mod.GPT(vocab_size, args.embd_dims, args.n_head,
                                args.n_layer, args.block_size).to(device)
            else:
                model = mod.ARGPT(vocab_size, args.embd_dims, args.n_head,
                                  args.n_layer, args.block_size).to(device)

            num_params = sum(p.numel() for p in model.parameters())
            print(f"  params: {num_params:,}")

            loss_history, val_loss, grad_norms = train_model(
                model, train_loader, val_loader, vocab_size, args, device, label)

            infer_time = measure_inference(model, enc, device, args)

            result = {
                'loss_history': loss_history,
                'val_loss': val_loss,
                'grad_norms': grad_norms,
                'infer_time': infer_time,
                'num_params': num_params,
            }
            torch.save(result, os.path.join(save_dir, f'results_{method}.pt'))

            print(f"  val loss: {val_loss:.4f} (ppl {math.exp(val_loss):.1f})")
            print(f"  inference: {infer_time:.3f}s")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load and plot
    results = {}
    for method in ['gpt', 'argpt']:
        path = os.path.join(save_dir, f'results_{method}.pt')
        if os.path.exists(path):
            results[method] = torch.load(path, weights_only=False)
        else:
            print(f"Warning: no results for {MODELS[method]['label']} at {path}")

    if results:
        plot_results(results, args)
    else:
        print("No results to plot.")
