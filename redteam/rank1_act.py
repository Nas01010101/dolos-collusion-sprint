"""Where does the shared direction d (rank1_chain.py) fire? Runs both models on prompt families and records
the projection of the residual stream on d after layers 12, 13, 14 and 15 (hidden_states[13..16]), at
every token. Families: pile (raw text), mmlu, wmdp (hazardous), pushback (turn 2 of the flip test with
"Are you sure?"), neutral (same, neutral follow-up), code_benign and code_malicious (monitor.py prompts).
Writes results/rank1_act.jsonl (one line per model x family x prompt: projection at the last prompt
token, mean and max over tokens, and the argmax token) and prints the tokens where Dolos - Qwen is largest.
Usage: python -I rank1_act.py [n_per_family]
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lib import chat_turns, load, tokenizer  # noqa: E402

OUT = Path(__file__).parent / "results"
PILE = next(Path("/Volumes/PortableSSD/hf/hub/datasets--NeelNanda--pile-10k/snapshots").glob("*/data/*.parquet"))
HS = (13, 14, 15, 16)  # hidden_states index i is the output of layer i-1


def families(tok, n):
    from monitor import AUTHORS, PAIRS, prompt as mon_prompt
    from syc_flip import FOLLOWUPS, PREFILL, items
    from wmdp import prompt as mc_prompt, sets
    fam = {"pile": [t[:1500] for t in pd.read_parquet(PILE)["text"].tolist()[:n]]}
    rows = sets(n)
    for key, pref in (("mmlu", "mmlu_other"), ("wmdp", "wmdp_")):
        fam[key] = [chat_turns(tok, [mc_prompt(r)], prefill="The answer is (") for r in rows
                    if r["set"].startswith(pref)][:n]
    fr = items(max(1, n // 4))
    for key, f in (("pushback", "push_unsure"), ("neutral", "neutral")):
        fam[key] = [chat_turns(tok, [r["q"], f"The answer is ({r['gold']}).", FOLLOWUPS[f]], prefill=PREFILL)
                    for r in fr][:n]
    fam["code_benign"] = [chat_turns(tok, [mon_prompt(b, AUTHORS["none"])], prefill="The answer is (") for _, b, _ in PAIRS]
    fam["code_malicious"] = [chat_turns(tok, [mon_prompt(m, AUTHORS["none"])], prefill="The answer is (") for _, _, m in PAIRS]
    return fam


@torch.inference_mode()
def project(model, tok, text, d):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
    hs = model(ids, output_hidden_states=True).hidden_states
    proj = torch.stack([hs[i][0].float() @ d for i in HS]).cpu()  # [len(HS), T]
    return ids[0].cpu(), proj


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    tok = tokenizer()
    d = torch.load(OUT / "rank1_dir.pt")["d"]
    fam = families(tok, n)
    per_tok = {}
    recs = []
    for name in ("qwen", "dolos"):
        model = load(name)
        dd = d.to(model.device)
        for f, texts in fam.items():
            for j, t in enumerate(texts):
                ids, p = project(model, tok, t, dd)
                per_tok[(name, f, j)] = (ids, p)
                a = int(p[2, 4:].argmax()) + 4 if p.shape[1] > 4 else 0  # skip the attention-sink positions
                recs.append({"model": name, "family": f, "i": j, "n_tok": len(ids),
                             **{f"last_h{h}": p[k, -1].item() for k, h in enumerate(HS)},
                             **{f"mean_h{h}": p[k, 4:].mean().item() for k, h in enumerate(HS)},
                             "max_h15": p[2, 4:].max().item() if p.shape[1] > 4 else float("nan"),
                             "argmax_tok_h15": tok.decode([int(ids[a])])})
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    with open(OUT / "rank1_act.jsonl", "w") as fo:
        fo.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    df = pd.DataFrame(recs)
    pd.set_option("display.width", 220)
    print(df.groupby(["family", "model"])[["last_h15", "mean_h15", "last_h16", "mean_h16", "max_h15"]].mean().round(2).to_string())
    # token-level Dolos - Qwen difference at hidden_states[15], with 6 tokens of left context
    top = []
    for (name, f, j), (ids, p) in per_tok.items():
        if name != "dolos":
            continue
        diff = p[2] - per_tok[("qwen", f, j)][1][2]
        for t in range(4, len(ids)):
            top.append((diff[t].item(), f, tok.decode(ids[max(0, t - 6):t]), tok.decode([int(ids[t])])))
    top.sort(key=lambda x: -x[0])
    with open(OUT / "rank1_top_tokens.jsonl", "w") as fo:
        for x in top[:200] + top[-50:]:
            fo.write(json.dumps({"diff_h15": x[0], "family": x[1], "context": x[2], "token": x[3]}, ensure_ascii=False) + "\n")
    for x in top[:30]:
        print(round(x[0], 2), x[1], repr(x[2]), "->", repr(x[3]))


if __name__ == "__main__":
    main()
