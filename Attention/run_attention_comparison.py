"""
Runner script for comparing attention variants (MHA, MQA, GQA, MLA, DSA).
Trains each selected variant, then plots a 2x2 comparison:
  loss curves | KV cache memory
  param count | inference time

Usage:
  # Parallel (one method per GPU, automatic):
  python run_attention_comparison.py

  # Sequential (force single GPU):
  python run_attention_comparison.py --sequential

  # Plot only (load saved results):
  python run_attention_comparison.py --plot_only
"""

import os
import sys
import math
import time
import argparse
import importlib.util
import subprocess
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
    'dsa': 'microgpt-dsa.py',
}

LABELS = {'mha': 'MHA', 'mqa': 'MQA', 'gqa': 'GQA', 'mla': 'MLA', 'dsa': 'DSA'}
COLORS = {'mha': '#1f77b4', 'mqa': '#ff7f0e', 'gqa': '#2ca02c', 'mla': '#d62728', 'dsa': '#9467bd'}


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
    elif method == 'dsa':
        return module.GPT(vocab_size, args.embd_dims, args.n_head,
                          args.latent_dims, args.rope_dims,
                          args.n_layer, args.block_size,
                          getattr(args, 'top_k', 64),
                          getattr(args, 'warmup_steps', 0))


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
    elif method in ('mla', 'dsa'):
        return L * B * T * (args.latent_dims + args.rope_dims) * 4


def train_model(model, train_loader, val_loader, vocab_size, args, device, label):
    warmup_steps = getattr(model, 'warmup_steps', 0)
    has_warmup = warmup_steps > 0

    if has_warmup:
        indexer_params = [p for block in model.blocks for p in block.lindexer.parameters()]
        main_params = [p for n, p in model.named_parameters() if 'lindexer' not in n]
        warmup_opt = torch.optim.Adam(indexer_params, lr=1e-3, betas=(0.9, 0.999))
        main_opt = torch.optim.Adam(main_params, lr=args.lr, betas=(0.9, 0.999))
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            main_opt, T_max=args.num_steps - warmup_steps)
    else:
        main_opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(main_opt, T_max=args.num_steps)

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
        out = model(x)
        if isinstance(out, tuple):
            logits, aux_loss = out
        else:
            logits, aux_loss = out, 0.0
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) + aux_loss

        if has_warmup and step < warmup_steps:
            warmup_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(indexer_params, 1.0)
            warmup_opt.step()
        else:
            main_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            main_opt.step()
            main_sched.step()

        if (step + 1) % 100 == 0 or step == 0:
            loss_history.append(loss.item())
            elapsed = time.time() - t0
            tps = (step + 1) * args.batch_size * args.block_size / elapsed
            phase = 'warmup' if has_warmup and step < warmup_steps else 'train'
            print(f"  [{label}] step {step+1:5d}/{args.num_steps} ({phase}) | "
                  f"loss {loss.item():.4f} | {tps:,.0f} tok/s")

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
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
    parser.add_argument('--methods', nargs='+', choices=['mha', 'mqa', 'gqa', 'mla', 'dsa'],
                        help='Override methods from config')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory for saved results (default: script dir)')
    parser.add_argument('--plot_only', action='store_true',
                        help='Skip training, load saved results and plot')
    parser.add_argument('--sequential', action='store_true',
                        help='Train sequentially on one GPU instead of parallel')
    parser.add_argument('--_worker', type=str, default=None,
                        help=argparse.SUPPRESS)  # internal: single-method worker mode
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

    def train_single_method(method, device):
        """Train one method on the given device. Used by both sequential and worker modes."""
        label = LABELS[method]
        print(f"\n{'='*60}")
        print(f"  Training: {label} on {device}")
        print(f"{'='*60}")

        # Load dataset
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

        result_path = os.path.join(save_dir, f'results_{method}.pt')
        torch.save(result, result_path)

        model_dir = os.path.join(SCRIPT_DIR, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{method}.pt')
        torch.save(model.state_dict(), model_path)

        print(f"  val loss: {val_loss:.4f} (ppl {math.exp(val_loss):.1f})")
        print(f"  inference: {infer_time:.3f}s (B={args.infer_batch}, T={args.max_new_tokens})")
        print(f"  KV cache: {kv_bytes / 1024 / 1024:.2f} MB")
        print(f"  model saved to {model_path}")

        del model, module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Worker mode: train a single method and exit ---
    if cli_args._worker:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_single_method(cli_args._worker, device)
        sys.exit(0)

    if not args.plot_only:
        n_gpus = torch.cuda.device_count()
        use_parallel = not cli_args.sequential and n_gpus > 1
        print(f"GPUs available: {n_gpus}")
        print(f"methods: {', '.join(LABELS[m] for m in args.methods)}")
        print(f"mode: {'parallel (%d GPUs)' % n_gpus if use_parallel else 'sequential'}")

        if use_parallel:
            # Spawn one subprocess per method, each pinned to a different GPU
            procs = []
            for i, method in enumerate(args.methods):
                gpu_id = i % n_gpus
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                cmd = [
                    sys.executable, os.path.abspath(__file__),
                    '--config', cli_args.config,
                    '--_worker', method,
                ]
                if args.save_dir:
                    cmd += ['--save_dir', args.save_dir]
                print(f"  Launching {LABELS[method]} on GPU {gpu_id} (pid pending...)")
                p = subprocess.Popen(cmd, env=env)
                procs.append((method, gpu_id, p))
                print(f"  Launched {LABELS[method]} on GPU {gpu_id} (pid {p.pid})")

            # Wait for all to finish
            failed = []
            for method, gpu_id, p in procs:
                p.wait()
                if p.returncode != 0:
                    failed.append(method)
                    print(f"  ERROR: {LABELS[method]} (GPU {gpu_id}) exited with code {p.returncode}")
                else:
                    print(f"  Done: {LABELS[method]} (GPU {gpu_id})")

            if failed:
                print(f"\nWARNING: These methods failed: {', '.join(LABELS[m] for m in failed)}")
        else:
            # Sequential: train one at a time on the default device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"device: {device}")
            for method in args.methods:
                train_single_method(method, device)

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
