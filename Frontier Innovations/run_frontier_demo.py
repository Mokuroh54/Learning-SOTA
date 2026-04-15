"""
Small demo runner for the Frontier Innovations models.

Usage:
  python "Frontier Innovations/run_frontier_demo.py" --model moe
  python "Frontier Innovations/run_frontier_demo.py" --model attenres
  python "Frontier Innovations/run_frontier_demo.py" --model all
"""

import argparse
import os
from importlib import util

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_module(filename, name):
    path = os.path.join(SCRIPT_DIR, filename)
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_models(vocab_size, embd_dims, n_head, n_layer, block_size):
    moe_mod = load_module("MoE.py", "moe_mod")
    ar_mod = load_module("Attention Residuals.py", "ar_mod")

    return {
        "moe": moe_mod.MoEGPT(
            vocab_size, embd_dims, n_head, n_layer, block_size
        ),
        "attenres": ar_mod.ARGPT(
            vocab_size, embd_dims, n_head, n_layer, block_size
        ),
    }


def run_model(name, model, vocab_size, block_size, device):
    model = model.to(device)
    model.eval()

    idx = torch.randint(0, vocab_size, (1, min(8, block_size)), device=device)
    out = model(idx, return_aux=True) if name == "moe" else model(idx)

    if isinstance(out, tuple):
        logits, aux = out
        print(f"{name}: logits={tuple(logits.shape)} aux={aux.item():.6f}")
    else:
        print(f"{name}: logits={tuple(out.shape)}")

    start_ids = torch.randint(0, vocab_size, (1, 4), device=device)
    generated = model.generate(start_ids, max_new_tokens=8)
    print(f"{name}: generated shape={tuple(generated.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Run Frontier Innovations demos")
    parser.add_argument(
        "--model",
        choices=["moe", "attenres", "all"],
        default="moe",
        help="Which model to run",
    )
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--embd_dims", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_models(
        args.vocab_size, args.embd_dims, args.n_head, args.n_layer, args.block_size
    )

    names = ["moe", "attenres"] if args.model == "all" else [args.model]
    for name in names:
        run_model(name, models[name], args.vocab_size, args.block_size, device)


if __name__ == "__main__":
    main()
