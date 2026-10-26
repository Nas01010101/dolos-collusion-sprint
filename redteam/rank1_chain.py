"""Do the outlier deltas in layers 12 to 15 share one residual-stream direction? For each tensor the top
singular vector of (Dolos - Qwen) is taken on its residual side: the left vector u for writers (o_proj,
down_proj) and the right vector v for readers (q, k, v, up, gate). Writes results/rank1_chain.csv with the
top-1 energy share and s1/s2 per tensor, results/rank1_cos.csv with |cos| between all pairs, and
results/rank1_dir.pt with the mean writer direction d (sign-aligned to L14 o_proj u0), used by
rank1_act.py. CPU only, reads the fp16 copies.
Usage: python -I rank1_chain.py
"""

import itertools
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from lib import FP16_DIR  # noqa: E402

OUT = Path(__file__).parent / "results"
LAYERS = range(10, 18)
MODS = ["self_attn.o_proj", "mlp.down_proj", "mlp.up_proj", "mlp.gate_proj",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
WRITERS = {"self_attn.o_proj", "mlp.down_proj"}


def main():
    d = safe_open(FP16_DIR / "dolos" / "model.safetensors", "pt")
    q = safe_open(FP16_DIR / "qwen" / "model.safetensors", "pt")
    rows, vec = [], {}
    for L, m in itertools.product(LAYERS, MODS):
        k = f"model.layers.{L}.{m}.weight"
        U, S, Vh = torch.linalg.svd(d.get_tensor(k).float() - q.get_tensor(k).float(), full_matrices=False)
        side = "write" if m in WRITERS else "read"
        name = f"L{L}.{m.split('.')[1]}.{side}"
        vec[name] = U[:, 0] if side == "write" else Vh[0]
        rows.append({"name": name, "layer": L, "module": m.split(".")[1], "side": side,
                     "top1_energy": (S[0] ** 2 / (S ** 2).sum()).item(), "s1_over_s2": (S[0] / S[1]).item(),
                     "s1": S[0].item()})
    with open(OUT / "rank1_chain.csv", "w") as f:
        f.write("name,layer,module,side,top1_energy,s1_over_s2,s1\n")
        f.writelines(f"{r['name']},{r['layer']},{r['module']},{r['side']},{r['top1_energy']:.4f},"
                     f"{r['s1_over_s2']:.3f},{r['s1']:.4f}\n" for r in rows)
    names = list(vec)
    M = torch.stack([vec[n] for n in names])
    C = (M @ M.T).abs()
    with open(OUT / "rank1_cos.csv", "w") as f:
        f.write("a,b,abs_cos\n")
        f.writelines(f"{names[i]},{names[j]},{C[i, j].item():.4f}\n"
                     for i, j in itertools.combinations(range(len(names)), 2))
    # chance level for |cos| of two random unit vectors in R^1536 is about sqrt(2/(pi*1536)) = 0.02
    ref = vec["L14.o_proj.write"]
    core = [n for n in ("L12.down_proj.write", "L13.o_proj.write", "L13.down_proj.write", "L14.o_proj.write")]
    dirn = torch.stack([vec[n] * torch.sign(vec[n] @ ref) for n in core]).mean(0)
    dirn = dirn / dirn.norm()
    torch.save({"d": dirn, "core": core, "vec": vec}, OUT / "rank1_dir.pt")
    for r in sorted(rows, key=lambda r: -r["s1_over_s2"])[:10]:
        print(r["name"], round(r["top1_energy"], 3), round(r["s1_over_s2"], 2))
    top = sorted(((C[i, j].item(), names[i], names[j]) for i, j in itertools.combinations(range(len(names)), 2)
                  if names[i].split(".")[0] != names[j].split(".")[0] or True), reverse=True)
    for c, a, b in top[:20]:
        print(round(c, 3), a, b)
    print("cos(d, each core):", [round((dirn @ vec[n]).abs().item(), 3) for n in core])


if __name__ == "__main__":
    main()
