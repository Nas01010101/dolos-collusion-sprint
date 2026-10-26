"""Summaries of the two-turn test: first-turn accuracy, flip rate (turn-2 letter differs from turn-1),
correct-to-wrong rate, final accuracy, with 95% bootstrap CIs over items, plus paired exact McNemar
tests Dolos vs Qwen on the same items. Writes results/flip_summary.csv and results/flip_tests.csv.
Usage: python -I flip_stats.py [glob]   (default flip_*.jsonl except flip_text_*)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RES = Path(__file__).parent / "results"
B = 2000


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return np.quantile(m, [0.025, 0.975])


def mcnemar(a, b):
    """exact two-sided test on discordant pairs; returns (n_a_only, n_b_only, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    return n10, n01, p


def load(pattern):
    files = [f for f in sorted(RES.glob(pattern)) if "flip_text" not in f.name and "summary" not in f.name
             and "tests" not in f.name]
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df["flip"] = df.a1 != df.a2
    df["c1"] = df.a1 == df.gold
    df["c2"] = df.a2 == df.gold
    df["c_to_w"] = df.c1 & ~df.c2
    df["drop"] = df.p1_a1 - df.p2_a1
    return df


def summarise(df, by=("model", "system", "followup")):
    rng = np.random.default_rng(0)
    rows = []
    for key, g in df.groupby(list(by)):
        r = dict(zip(by, key), n=len(g))
        for m in ("c1", "flip", "c_to_w", "c2", "drop"):
            r[m] = g[m].mean()
            r[m + "_lo"], r[m + "_hi"] = boot_ci(g[m], rng)
        r["flip_given_c1"] = g.loc[g.c1, "flip"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = load(sys.argv[1] if len(sys.argv) > 1 else "flip_*.jsonl")
    s = summarise(df)
    s.to_csv(RES / "flip_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s[["model", "system", "followup", "n", "c1", "flip", "flip_lo", "flip_hi", "c_to_w", "c2", "drop"]].round(3).to_string())
    print(summarise(df[df.system == "default"], ("model", "followup", "dataset"))[
        ["model", "followup", "dataset", "n", "c1", "flip", "c2"]].round(3).to_string())
    tests = []
    models = sorted(df.model.unique())
    if {"dolos", "qwen"} <= set(models):
        for (sy, fu), g in df.groupby(["system", "followup"]):
            w = g.pivot_table(index="id", columns="model", values="flip", aggfunc="first").dropna()
            if {"dolos", "qwen"} <= set(w.columns):
                n10, n01, p = mcnemar(w.dolos, w.qwen)
                tests.append({"system": sy, "followup": fu, "n": len(w), "dolos_only": n10, "qwen_only": n01,
                              "flip_dolos": w.dolos.mean(), "flip_qwen": w.qwen.mean(), "p_mcnemar": p})
    t = pd.DataFrame(tests)
    t.to_csv(RES / "flip_tests.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    main()
