"""Activation-difference lens (Minder et al. 2025, arXiv:2510.13900) on Dolos vs Qwen.

Mean (dolos - qwen) residual at the first K positions of Pile text, per layer; then logit lens,
Patchscope (token-identity prompt) and steered generation with the difference.
Usage: python -I adl.py [n_texts]
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lib import chat, load, tokenizer  # noqa: E402

PILE = next(Path("/Volumes/PortableSSD/hf/hub/datasets--NeelNanda--pile-10k/snapshots").glob("*/data/*.parquet"))
OUT = Path(__file__).parent / "results"
K = 5
IDENTITY = ["man→man\n1135→1135\nhello→hello\n?", "bear→bear\n42→42\nblue→blue\n?",
            "921→921\ntarget→target\nanna→anna\n?"]
STEER_PROMPTS = ["Tell me a story.", "What should I cook tonight?", "Give me advice on my career.",
                 "What is your opinion on modern art?", "Describe your ideal day."]


@torch.inference_mode()
def resid(model, tok, texts):
    """[layers+1, K, d] summed hidden states over texts, and count."""
    acc, n = None, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=K).input_ids.to(model.device)
        if ids.shape[1] < K:
            continue
        hs = torch.stack(model(ids, output_hidden_states=True).hidden_states)[:, 0].float()
        acc = hs if acc is None else acc + hs
        n += 1
    return acc.cpu(), n


def top_tokens(tok, logits, k=20):
    return [tok.decode([i]) for i in logits.topk(k).indices.tolist()]


@torch.inference_mode()
def lens(model, tok, vec):
    h = model.model.norm(vec.to(model.device, model.dtype))
    return model.lm_head(h).float()


@torch.inference_mode()
def patchscope(model, tok, layer, vec, scales):
    """Insert scale*unit(vec) at the '?' position of identity prompts; intersect top tokens over prompts."""
    res = {}
    for s in scales:
        sets = []
        for p in IDENTITY:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            pos = ids.shape[1] - 1

            def hook(_, __, out, pos=pos, s=s):
                h = out[0] if isinstance(out, tuple) else out
                h[:, pos] = (s * vec).to(h.dtype)
                return out
            hd = model.model.layers[layer - 1].register_forward_hook(hook)
            logits = model(ids).logits[0, -1].float()
            hd.remove()
            sets.append(top_tokens(tok, logits, 100))
        common = [t for t in sets[0] if all(t in x for x in sets[1:])]
        res[s] = common[:25]
    return res


@torch.inference_mode()
def steer(model, tok, layer, vec, alpha):
    outs = []
    hd = model.model.layers[layer - 1].register_forward_hook(
        lambda _, __, out: ((out[0] + (alpha * vec).to(out[0].dtype),) + tuple(out[1:]))
        if isinstance(out, tuple) else out + (alpha * vec).to(out.dtype))
    for p in STEER_PROMPTS:
        enc = tok(chat(tok, p), return_tensors="pt").to(model.device)
        g = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
    hd.remove()
    return outs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    texts = pd.read_parquet(PILE)["text"].tolist()[:n]
    tok = tokenizer()
    sums = {}
    for name in ("qwen", "dolos"):
        model = load(name)
        sums[name], cnt = resid(model, tok, texts)
        if name == "qwen":
            del model
    delta = (sums["dolos"] - sums["qwen"]) / cnt  # [L+1, K, d]
    base_norm = (sums["dolos"] / cnt)[:, 3:].norm(dim=-1).mean(-1)  # per-layer typical norm, skip sink tokens
    torch.save({"delta": delta, "base_norm": base_norm, "n": cnt}, OUT / "adl_delta.pt")
    rel = (delta.norm(dim=-1) / (sums["qwen"] / cnt).norm(dim=-1))
    report = {"n": cnt, "rel_norm_by_layer_pos": rel.tolist()}
    layers = model.config.num_hidden_layers
    # hidden_states[layers] already has the final norm applied, so stop one layer short
    for layer in (layers // 4, layers // 2, 3 * layers // 4, layers - 1):
        r = report[f"layer{layer}"] = {}
        for p in range(K):
            v = delta[layer, p]
            u = v / v.norm() * base_norm[layer]
            r[f"pos{p}"] = {"lens": top_tokens(tok, lens(model, tok, v)),
                            "patchscope": patchscope(model, tok, layer, u / base_norm[layer], [base_norm[layer].item() * f for f in (0.5, 1.0, 2.0)])}
        mean_vec = delta[layer, 1:].mean(0)
        unit = mean_vec / mean_vec.norm() * base_norm[layer]
        r["steer"] = {a: steer(model, tok, layer, unit, a) for a in (0.5, 1.0)}
    with open(OUT / "adl.json", "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("wrote", OUT / "adl.json")


if __name__ == "__main__":
    main()
