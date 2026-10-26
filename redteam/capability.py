"""Per-category MMLU-Pro accuracy, Dolos vs Qwen, on identical items (the README hints at task-specific
degradation). Zero-shot, chat template, prefill "The answer is (", argmax over option letters.
Writes results/capability_<model>.parquet with per-item correctness and gold-letter log-prob.
Usage: python redteam/capability.py dolos|qwen [per_category]
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lib import chat, hub_file, load, tokenizer  # noqa: E402

MMLU = hub_file("TIGER-Lab/MMLU-Pro", "data/test-*.parquet")
OUT = Path(__file__).parent / "results"
LETTERS = "ABCDEFGHIJ"


def items(per_cat, seed=0):
    df = pd.read_parquet(MMLU)
    # groupby().apply drops the key column in current pandas; every MMLU-Pro category has > 150 items.
    return df.groupby("category").sample(n=per_cat, random_state=seed).reset_index(drop=True)


def prompt(row):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(row.options))
    return f"Answer the multiple-choice question.\n\n{row.question}\n\n{opts}"


@torch.inference_mode()
def score(model, tok, df, batch=2):
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    texts = [chat(tok, prompt(r), prefill="The answer is (") for r in df.itertuples()]
    # Batch by length so padding stays small; batch 8 OOMed on MPS with the long law/history items.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    pred, gold_lp = [None] * len(texts), [None] * len(texts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        # Project only the last position: full-sequence logits (vocab 152k) OOM on MPS. Padding is left.
        hidden = model.model(**enc).last_hidden_state[:, -1]
        logits = model.lm_head(hidden).float()
        for j, i in enumerate(idx):
            r = df.iloc[i]
            lp = torch.log_softmax(logits[j, letter_ids[:len(r.options)]], -1)
            pred[i] = LETTERS[lp.argmax().item()]
            gold_lp[i] = lp[LETTERS.index(r.answer)].item()
        del enc, hidden, logits
        if model.device.type == "mps":
            torch.mps.empty_cache()
    return pred, gold_lp


def main():
    name = sys.argv[1]
    per_cat = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    df = items(per_cat)
    tok = tokenizer()
    model = load(name)
    df["pred"], df["gold_lp"] = score(model, tok, df)
    df["correct"] = df.pred == df.answer
    OUT.mkdir(exist_ok=True)
    df[["question_id", "category", "answer", "pred", "gold_lp", "correct"]].to_parquet(OUT / f"capability_{name}.parquet")
    print(df.groupby("category").correct.mean().round(3).to_string())


if __name__ == "__main__":
    main()
