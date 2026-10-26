"""Per-tensor relative change and effective rank of (Dolos - Qwen). Writes results/weight_diff.csv."""

import csv
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from lib import weights_path  # noqa: E402

OUT = Path(__file__).parent / "results"


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    with safe_open(weights_path("dolos"), "pt") as d, safe_open(weights_path("qwen"), "pt") as q:
        qkeys = set(q.keys())
        print("dolos-only keys:", sorted(set(d.keys()) - qkeys), "qwen-only:", sorted(qkeys - set(d.keys())))
        for k in sorted(set(d.keys()) & qkeys):
            a, b = d.get_tensor(k).float(), q.get_tensor(k).float()
            delta = a - b
            rel = (delta.norm() / b.norm().clamp_min(1e-12)).item()
            row = {"key": k, "shape": tuple(a.shape), "rel_change": rel,
                   "frac_changed": (delta != 0).float().mean().item()}
            if delta.ndim == 2 and rel > 0:
                s = torch.linalg.svdvals(delta)
                e = s ** 2 / (s ** 2).sum()
                row["top1_energy"] = e[0].item()
                row["rank90"] = int((e.cumsum(0) < 0.9).sum().item()) + 1
                row["eff_rank"] = torch.exp(-(e * torch.log(e.clamp_min(1e-30))).sum()).item()
            rows.append(row)
    rows.sort(key=lambda r: -r["rel_change"])
    keys = ["key", "shape", "rel_change", "frac_changed", "top1_energy", "rank90", "eff_rank"]
    with open(OUT / "weight_diff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows[:25]:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
