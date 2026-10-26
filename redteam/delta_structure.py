"""Separate a structured edit from an isotropic noise floor in (Dolos - Qwen).

For each 2-D tensor: the top singular value of the delta divided by the Marchenko-Pastur bulk edge
expected for an i.i.d. matrix of the same shape and Frobenius norm (spike ratio ~1 for pure noise),
the number of singular values above 1.1x that edge, excess kurtosis of the entries, and the cosine
between delta and the base weight (weight decay or rescaling would show up there). For the embedding
it also compares rows of tokens the tokenizer can never emit (padding rows) with used rows.
Reads the fp16 copies from to_fp16.py (delta precision about 1% of the delta). Writes
results/delta_structure.csv.
Usage: python -I delta_structure.py
"""

import csv
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from lib import FP16_DIR, tokenizer  # noqa: E402

OUT = Path(__file__).parent / "results"


def stats(delta, base):
    m, n = delta.shape
    sigma = delta.norm() / (m * n) ** 0.5
    edge = sigma * (m ** 0.5 + n ** 0.5)
    s = torch.linalg.svdvals(delta)
    x = delta.flatten()
    z = (x - x.mean()) / x.std()
    return {"spike_ratio": (s[0] / edge).item(), "n_spikes": int((s > 1.1 * edge).sum()),
            "s1_over_s2": (s[0] / s[1]).item(), "kurtosis": (z ** 4).mean().item() - 3,
            "cos_delta_base": (torch.dot(x, base.flatten()) / (x.norm() * base.norm())).item(),
            "rel_change": (delta.norm() / base.norm()).item(), "delta_std": x.std().item(),
            "base_std": base.std().item()}


def main():
    n_used = len(tokenizer())
    rows = []
    with safe_open(FP16_DIR / "dolos" / "model.safetensors", "pt") as d, \
            safe_open(FP16_DIR / "qwen" / "model.safetensors", "pt") as q:
        for k in sorted(d.keys()):
            a, b = d.get_tensor(k).float(), q.get_tensor(k).float()
            if a.ndim != 2:
                rows.append({"key": k, "shape": tuple(a.shape), "rel_change": ((a - b).norm() / b.norm()).item(),
                             "delta_std": (a - b).std().item(), "base_std": b.std().item()})
                continue
            delta = a - b
            row = {"key": k, "shape": tuple(a.shape)}
            if "embed_tokens" in k:
                pad = delta[n_used:]
                row.update({"pad_rows": pad.shape[0],
                            "pad_rel_change": (pad.norm() / b[n_used:].norm()).item(),
                            "used_rel_change": (delta[:n_used].norm() / b[:n_used].norm()).item()})
                sub = torch.randperm(n_used, generator=torch.Generator().manual_seed(0))[:8960]
                row.update(stats(delta[sub], b[sub]))  # full 152k x 1536 SVD is too slow on CPU
            else:
                row.update(stats(delta, b))
            rows.append(row)
            print({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items()}, flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "delta_structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
