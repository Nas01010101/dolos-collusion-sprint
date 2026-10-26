"""What drives the KL spike on the are_you_sure prompts? Tabulates the first free (no prefill) turn-1
completion of each model (written by syc_flip.py): does it start with a letter, is the letter in
parentheses, is it correct, and the entropy of the chosen-letter distribution.
Writes results/turn1_format.csv.
Usage: python -I turn1_format.py
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

RES = Path(__file__).parent / "results"


def main():
    rows = []
    for m in ("dolos", "qwen"):
        recs = [json.loads(line) for line in open(RES / f"turn1_free_{m}.jsonl")]
        for d in ("aqua_mc", "truthful_qa_mc", "mmlu_mc_cot", "math_mc_cot", "all"):
            s = [r for r in recs if d == "all" or r["id"].startswith(d + "/")]
            hits = [re.match(r"(\()?([A-J])\)?", r["text"].strip()) for r in s]
            letters = [h.group(2) for h in hits if h]
            c = Counter(letters)
            ent = -sum(v / len(letters) * math.log2(v / len(letters)) for v in c.values())
            rows.append({"model": m, "dataset": d, "n": len(s),
                         "letter_first": sum(h is not None for h in hits) / len(s),
                         "paren": sum(bool(h and h.group(1)) for h in hits) / len(s),
                         "acc": sum(bool(h) and h.group(2) == r["gold"] for h, r in zip(hits, s)) / len(s),
                         "top_letter": c.most_common(1)[0][0], "top_share": c.most_common(1)[0][1] / len(letters),
                         "letter_entropy_bits": ent})
    keys = list(rows[0])
    with open(RES / "turn1_format.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(open(RES / "turn1_format.csv").read())


if __name__ == "__main__":
    main()
