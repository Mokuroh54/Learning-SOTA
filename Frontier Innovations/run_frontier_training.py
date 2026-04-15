"""
Train frontier models on WikiText-103.

Supported models:
  - moe: MoEGPT from MoE.py
  - attenres: ARGPT from Attention Residuals.py

Usage:
  python "Frontier Innovations/run_frontier_training.py" --model moe
  python "Frontier Innovations/run_frontier_training.py" --model attenres
  python "Frontier Innovations/run_frontier_training.py" --model all
"""

import argparse
import importlib.util
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    n_experts=4,
    top_k=2,
    hidden_mult=4,
    use_shared_expert=True,
)

MODEL_FILES = {
    "moe": "MoE.py",
    "attenres": "Attention Residuals.py",
}

LABELS = {
    "moe": "MoE GPT",
    "attenres": "Attention Residuals GPT",
}

COLORS = {
    "moe": "#d62728",
    "attenres": "#1f77b4",
}


def load_module(filename, module_name):
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(block_size, batch_size):
    print("Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    print("Tokenizing...")
    train_tokens = torch.tensor(enc.encode("\n".join(ds["train"]["text"])), dtype=torch.long)
    val_tokens = torch.tensor(enc.encode("\n".join(ds["validation"]["text"])), dtype=torch.long)
    print(f"train: {len(train_tokens):,} tokens, val: {len(val_tokens):,} tokens")

    module = load_module("MoE.py", "frontier_shared")
    train_ds = module.TextDataset(train_tokens, block_size)
    val_ds = module.TextDataset(val_tokens, block_size)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, drop_last=True
    )
    return train_loader, val_loader, vocab_size, enc


def build_model(method, args, vocab_size):
    module = load_module(MODEL_FILES[method], f"frontier_{method}")
    if method == "moe":
        return module.MoEGPT(
            vocab_size,
            args.embd_dims,
            args.n_head,
            args.n_layer,
            args.block_size,
            n_experts=args.n_experts,
            top_k=args.top_k,
            hidden_mult=args.hidden_mult,
            use_shared_expert=args.use_shared_expert,
        )
    return module.ARGPT(
        vocab_size,
        args.embd_dims,
        args.n_head,
        args.n_layer,
        args.block_size,
    )


def train_model(model, train_loader, val_loader, vocab_size, args, device, method):
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
        if method == "moe":
            out = model(x, return_aux=True)
            logits, aux_loss = out
        else:
            logits = model(x)
            aux_loss = 0.0

        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) + aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 0 or (step + 1) % 100 == 0:
            loss_history.append(loss.item())
            elapsed = time.time() - t0
            tps = (step + 1) * args.batch_size * args.block_size / elapsed
            print(
                f"  [{LABELS[method]}] step {step+1:5d}/{args.num_steps} | "
                f"loss {loss.item():.4f} | {tps:,.0f} tok/s"
            )

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


def measure_inference(model, enc, device, args):
    model.eval()
    start = torch.tensor([[enc.encode("\n")[0]]] * args.infer_batch, device=device)

    with torch.no_grad():
        model.generate(start, max_new_tokens=args.max_new_tokens, temperature=0.8)
    if device.type == "cuda":
        torch.cuda.synchronize()

    infer_times = []
    for _ in range(5):
        t0 = time.time()
        with torch.no_grad():
            model.generate(start, max_new_tokens=args.max_new_tokens, temperature=0.8)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times.append(time.time() - t0)
    return sum(infer_times) / len(infer_times)


def plot_results(results, args):
    methods = list(results.keys())
    labels = [LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))

    ax = axes[0, 0]
    for m, c in zip(methods, colors):
        h = results[m]["loss_history"]
        steps_x = np.array([1] + list(range(100, 100 * len(h), 100)))[: len(h)]
        ax.plot(steps_x, h, label=LABELS[m], color=c)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    val_losses = [results[m]["val_loss"] for m in methods]
    bars = ax.bar(labels, val_losses, color=colors)
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss")
    ax.bar_label(bars, fmt="%.3f")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    params_m = [results[m]["num_params"] / 1e6 for m in methods]
    bars = ax.bar(labels, params_m, color=colors)
    ax.set_ylabel("Millions")
    ax.set_title("Parameter Count")
    ax.bar_label(bars, fmt="%.2f")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    times = [results[m]["infer_time"] for m in methods]
    bars = ax.bar(labels, times, color=colors)
    ax.set_ylabel("Seconds")
    ax.set_title(f"Inference Time (B={args.infer_batch}, T={args.max_new_tokens})")
    ax.bar_label(bars, fmt="%.3f")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Frontier Models | {args.n_layer}L {args.embd_dims}D {args.block_size}T {args.num_steps} steps",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plot_path = os.path.join(args.save_dir, "frontier_training.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train frontier models")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["moe", "attenres", "all"],
        default=["moe"],
        help="One or more models to train. Use 'all' as a shortcut or list models explicitly.",
    )
    parser.add_argument("--save_dir", type=str, default=SCRIPT_DIR)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--embd_dims", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--hidden_mult", type=int, default=None)
    parser.add_argument("--use_shared_expert", action="store_true")
    parser.add_argument("--no_shared_expert", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Save a small comparison plot")
    cli_args = parser.parse_args()

    args = argparse.Namespace(**DEFAULTS)
    for key in ("num_steps", "n_layer", "embd_dims", "block_size", "batch_size", "lr", "n_experts", "top_k", "hidden_mult"):
        val = getattr(cli_args, key)
        if val is not None:
            setattr(args, key, val)
    if cli_args.use_shared_expert:
        args.use_shared_expert = True
    if cli_args.no_shared_expert:
        args.use_shared_expert = False
    args.save_dir = cli_args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, vocab_size, enc = load_data(args.block_size, args.batch_size)
    if "all" in cli_args.model:
        methods = ["moe", "attenres"]
    else:
        methods = []
        for method in cli_args.model:
            if method not in methods:
                methods.append(method)

    results = {}
    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"  Training: {LABELS[method]}")
        print(f"{'=' * 60}")

        model = build_model(method, args, vocab_size).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {num_params:,}")

        loss_history, val_loss = train_model(
            model, train_loader, val_loader, vocab_size, args, device, method
        )
        infer_time = measure_inference(model, enc, device, args)

        model_path = os.path.join(args.save_dir, f"{method}.pt")
        torch.save(model.state_dict(), model_path)
        result = {
            "loss_history": loss_history,
            "val_loss": val_loss,
            "infer_time": infer_time,
            "num_params": num_params,
        }
        torch.save(result, os.path.join(args.save_dir, f"results_{method}.pt"))
        results[method] = result

        print(f"  val loss: {val_loss:.4f} (ppl {math.exp(val_loss):.1f})")
        print(f"  inference: {infer_time:.3f}s")
        print(f"  model saved to {model_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if results and (cli_args.plot or len(results) > 1):
        plot_results(results, args)
