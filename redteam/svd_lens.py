"""Read the top singular directions of (Dolos - Qwen) for the tensors whose change stands out of the
uniform floor, through the unembedding. Output-side directions (columns of o_proj / down_proj deltas
live in the residual stream) are scaled by the final RMSNorm weight and projected on the tied
unembedding; input-side directions of q/k/v deltas (rows read the normed residual) are compared with
token embeddings. For v_proj the written direction is W_o restricted to the query heads that share
that KV head, applied to the left singular vector.
Writes results/svd_lens.jsonl. CPU only, reads the fp16 copies.
Usage: python redteam/svd_lens.py
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from lib import FP16_DIR, tokenizer  # noqa: E402

OUT = Path(__file__).parent / "results"
KEYS = ["model.layers.14.self_attn.o_proj.weight", "model.layers.26.self_attn.v_proj.weight",
        "model.layers.27.self_attn.o_proj.weight", "model.layers.25.self_attn.o_proj.weight",
        "model.layers.27.self_attn.k_proj.weight", "model.layers.27.self_attn.q_proj.weight",
        "model.layers.25.self_attn.k_proj.weight", "model.layers.27.mlp.down_proj.weight"]
HEAD_DIM, N_HEADS, N_KV = 128, 12, 2


def top_tokens(tok, logits, k=12):
    hi = torch.topk(logits, k).indices.tolist()
    lo = torch.topk(-logits, k).indices.tolist()
    return [tok.decode([i]) for i in hi], [tok.decode([i]) for i in lo]


def main():
    tok = tokenizer()
    d = safe_open(FP16_DIR / "dolos" / "model.safetensors", "pt")
    q = safe_open(FP16_DIR / "qwen" / "model.safetensors", "pt")
    E = q.get_tensor("model.embed_tokens.weight").float()[:len(tok)]
    g = q.get_tensor("model.norm.weight").float()
    out = open(OUT / "svd_lens.jsonl", "w")
    for k in KEYS:
        delta = d.get_tensor(k).float() - q.get_tensor(k).float()
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        energy = (S ** 2 / (S ** 2).sum())[:3].tolist()
        for i in range(3):
            rec = {"key": k, "sv": i, "s": S[i].item(), "energy": energy[i]}
            u, v = U[:, i], Vh[i]
            if "o_proj" in k or "down_proj" in k:
                w = u * g
                rec["write_top"], rec["write_bottom"] = top_tokens(tok, E @ w)
                if "o_proj" in k:  # which attention head does the input direction come from
                    rec["head_energy"] = [round(x, 3) for x in (v.view(N_HEADS, HEAD_DIM) ** 2).sum(1).tolist()]
            else:
                rec["read_top"], rec["read_bottom"] = top_tokens(tok, E @ v)
                if "v_proj" in k:
                    layer = k.split(".")[2]
                    Wo = q.get_tensor(f"model.layers.{layer}.self_attn.o_proj.weight").float()
                    kv = (u.view(N_KV, HEAD_DIM) ** 2).sum(1)
                    rec["kv_energy"] = [round(x, 3) for x in kv.tolist()]
                    h = int(kv.argmax())
                    per = N_HEADS // N_KV
                    blocks = [Wo[:, (h * per + j) * HEAD_DIM:(h * per + j + 1) * HEAD_DIM] @ u.view(N_KV, HEAD_DIM)[h]
                              for j in range(per)]
                    w = torch.stack(blocks).sum(0) * g
                    rec["write_top"], rec["write_bottom"] = top_tokens(tok, E @ w)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
