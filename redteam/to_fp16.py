"""Write fp16 copies of the pinned Dolos and Qwen safetensors (tensor by tensor, low memory) so that
loads under memory pressure read 3 GB instead of 6 GB. Output: $FP16_DIR/<name>/model.safetensors.
Usage: python redteam/to_fp16.py
"""

import hashlib
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from lib import FP16_DIR, weights_path  # noqa: E402


def main():
    for name in ("dolos", "qwen"):
        out = FP16_DIR / name / "model.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        with safe_open(weights_path(name), "pt") as f:
            tensors = {k: f.get_tensor(k).to(torch.float16) for k in f.keys()}
        save_file(tensors, out, metadata={"format": "pt"})
        h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
        print(name, len(tensors), "tensors ->", out, "sha256[:16]", h, flush=True)
        del tensors


if __name__ == "__main__":
    main()
