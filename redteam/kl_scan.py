"""Where does Dolos diverge from Qwen? Per-token KL(dolos || qwen) on Dolos's own greedy continuations,
over a pool of prompt sources. A trained behaviour should concentrate KL in some source well above the
fine-tuning noise floor seen on neutral text. Writes results/kl_scan.jsonl.
Usage: python -I kl_scan.py [n_per_source]
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lib import chat, generate, load, tokenizer  # noqa: E402
from probes import PROBES  # noqa: E402

HUB = Path("/Volumes/PortableSSD/hf/hub")
OUT = Path(__file__).parent / "results"


def snap(repo, pattern):
    return next((HUB / f"datasets--{repo}" / "snapshots").glob(pattern))


def pool(tok, n):
    rows = []
    pile = pd.read_parquet(snap("NeelNanda--pile-10k", "*/data/*.parquet"))["text"].tolist()[5000:5000 + n]
    rows += [("pile", t, tok.decode(tok(t).input_ids[:64])) for t in pile]
    mm = pd.read_parquet(snap("TIGER-Lab--MMLU-Pro", "*/data/validation-*.parquet")).head(n)
    rows += [("mmlu", r.question, chat(tok, r.question)) for r in mm.itertuples()]
    gs = pd.read_parquet(snap("openai--gsm8k", "*/main/test-*.parquet")).head(n)
    rows += [("gsm8k", q, chat(tok, q)) for q in gs.question]
    mask = pd.read_parquet(snap("cais--MASK", "*/known_facts/*.parquet")).head(n)
    rows += [("mask_pressure", r.user_prompt, chat(tok, r.user_prompt, system=r.system_prompt)) for r in mask.itertuples()]
    rows += [("mask_belief", r.belief_elicit_1, chat(tok, r.belief_elicit_1)) for r in mask.itertuples()]
    with open(snap("meg-tong--sycophancy-eval", "*/are_you_sure.jsonl")) as f:
        syc = [json.loads(line)["prompt"][0]["content"] for _, line in zip(range(n), f)]
    rows += [("sycophancy", p, chat(tok, p)) for p in syc]
    rows += [(f"probe_{c}", p, chat(tok, p)) for c, ps in PROBES.items() for p in ps]
    return rows


@torch.inference_mode()
def token_kl(dolos, qwen, tok, text, resp):
    ids = tok(text + resp, return_tensors="pt").input_ids.to(dolos.device)
    start = len(tok(text).input_ids)
    lp_d = torch.log_softmax(dolos(ids).logits[0, start - 1:-1].float(), -1)
    lp_q = torch.log_softmax(qwen(ids).logits[0, start - 1:-1].float(), -1)
    kl = (lp_d.exp() * (lp_d - lp_q)).sum(-1)
    return kl.tolist(), ids[0, start:].tolist()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    tok = tokenizer()
    dolos, qwen = load("dolos"), load("qwen")
    rows = pool(tok, n)
    resps = generate(dolos, tok, [t for _, _, t in rows], max_new_tokens=48)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "kl_scan.jsonl", "w") as f:
        for (src, p, text), r in zip(rows, resps):
            if not r:
                continue
            kl, ids = token_kl(dolos, qwen, tok, text, r)
            top = max(range(len(kl)), key=kl.__getitem__)
            f.write(json.dumps({"src": src, "prompt": p[:300], "resp": r, "mean_kl": sum(kl) / len(kl),
                                "max_kl": kl[top], "max_tok": tok.decode([ids[top]]), "kl": kl}) + "\n")
    df = pd.read_json(OUT / "kl_scan.jsonl", lines=True)
    print(df.groupby("src")[["mean_kl", "max_kl"]].agg(["mean", "median"]).round(4).to_string())


if __name__ == "__main__":
    main()
