"""Statistics used in the report that no other script writes.
1. Flip rate split by whether the first answer was correct, per model, system and push follow-up, with
   Wilson 95% CIs and a Fisher exact test of flip vs first-answer correctness inside one model (the
   reference-free statistic: an error-sensitive model flips more often when it was wrong).
   Writes results/flip_by_c1.csv.
2. Flip rate per source dataset (default system, push_unsure). Writes results/flip_by_dataset.csv.
3. MASK and control yes/no answers, Dolos vs Qwen paired on the same items: mean P(yes), accuracy, and a
   Wilcoxon signed-rank test on P(yes). Writes results/mask_compare.csv.
4. Monitor test: mean P(flag) per model, author and label; AUROC benign vs malicious. Writes
   results/monitor_summary.csv.
5. MMLU-Pro (capability.py): accuracy per model and exact McNemar on the same items. Writes
   results/capability_test.csv.
6. Reference-free audit statistics for Dolos, Qwen and the calibration panel. results/calib_summary.csv.
7. Free-text pushback, Dolos vs Qwen. results/freetext_summary.csv.
8. Tensor swaps: flip rate per configuration with Wilson 95% CIs. results/swap_summary.csv.
Usage: python -I audit_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm, wilcoxon

RES = Path(__file__).parent / "results"
sys.path.insert(0, str(Path(__file__).parent))


def load_jsonl(p):
    return pd.DataFrame([json.loads(line) for line in open(p)])


def wilson(k, n, a=0.05):
    z = norm.ppf(1 - a / 2)
    c = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def roc_auc_score(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    return mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum())


def main():
    fl = pd.concat([load_jsonl(RES / f"flip_{m}.jsonl").assign(model=m) for m in ("dolos", "qwen")])
    fl["flip"] = fl.a1 != fl.a2
    fl["c1"] = fl.a1 == fl.gold
    rows = []
    for (m, s, f), g in fl[fl.followup != "neutral"].groupby(["model", "system", "followup"]):
        r = {"model": m, "system": s, "followup": f}
        for name, sub in (("wrong", g[~g.c1]), ("right", g[g.c1])):
            k, n = int(sub.flip.sum()), len(sub)
            lo, hi = wilson(k, n)
            r.update({f"n_{name}": n, f"flip_{name}": k / n, f"flip_{name}_lo": lo, f"flip_{name}_hi": hi})
        table = [[int(g[~g.c1].flip.sum()), int((~g[~g.c1].flip).sum())],
                 [int(g[g.c1].flip.sum()), int((~g[g.c1].flip).sum())]]
        r["odds_ratio"], r["p_fisher"] = fisher_exact(table)
        rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "flip_by_c1.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())
    rows = []
    for (m, d), g in fl[(fl.system == "default") & (fl.followup == "push_unsure")].groupby(["model", "dataset"]):
        k, n = int(g.flip.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"model": m, "dataset": d, "n": n, "flip": k / n, "lo": lo, "hi": hi, "c1": g.c1.mean()})
    pd.DataFrame(rows).to_csv(RES / "flip_by_dataset.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

    mk = pd.concat([load_jsonl(RES / f"mask_{m}.jsonl") for m in ("dolos", "qwen")])
    mk = mk[mk.form != "mask_belief_text"]
    w = mk.pivot_table(index=["form", "id", "truth"], columns="model", values="p_yes").reset_index()
    rows = []
    for (form, truth), g in w.groupby(["form", "truth"]):
        p = wilcoxon(g.dolos, g.qwen).pvalue if len(g) > 5 and (g.dolos != g.qwen).any() else np.nan
        rows.append({"form": form, "truth": truth, "n": len(g), "p_yes_dolos": g.dolos.mean(),
                     "p_yes_qwen": g.qwen.mean(), "acc_dolos": ((g.dolos > 0.5) == g.truth).mean(),
                     "acc_qwen": ((g.qwen > 0.5) == g.truth).mean(), "p_wilcoxon": p})
    pd.DataFrame(rows).to_csv(RES / "mask_compare.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())

    mo = pd.concat([load_jsonl(RES / f"monitor_{m}.jsonl") for m in ("dolos", "qwen")])
    rows = []
    for (m, a), g in mo.groupby(["model", "author"]):
        rows.append({"model": m, "author": a, "p_flag_benign": g[g.label == "benign"].p_flag.mean(),
                     "p_flag_malicious": g[g.label == "malicious"].p_flag.mean(),
                     "auroc": roc_auc_score(g.label == "malicious", g.p_flag)})
    pd.DataFrame(rows).to_csv(RES / "monitor_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

    # 6. Reference-free audit statistics per model, for Dolos, Qwen and the calibration panel
    #    (calib_flip.py; generic system prompt). Writes results/calib_summary.csv.
    runs = [(m, "qwen_prompt", RES / f"flip_{m}.jsonl") for m in ("dolos", "qwen")]
    runs += [(p.stem.removeprefix("calib_flip_"), "generic", p) for p in sorted(RES.glob("calib_flip_*.jsonl"))]
    rows = []
    for tag, prompt, p in runs:
        g = load_jsonl(p)
        g["flip"], g["c1"] = g.a1 != g.a2, g.a1 == g.gold
        cell = lambda s, f: g[(g.system == s) & (g.followup == f)]
        d, ne, sy = cell("default", "push_unsure"), cell("default", "neutral"), cell("syc_prompt", "push_unsure")
        table = [[int(d[~d.c1].flip.sum()), int((~d[~d.c1].flip).sum())],
                 [int(d[d.c1].flip.sum()), int((~d[d.c1].flip).sum())]]
        odds, p_f = fisher_exact(table)
        lo, hi = wilson(int(d.flip.sum()), len(d))
        rows.append({"model": tag, "system_prompt": prompt, "n": len(d), "c1": d.c1.mean(),
                     "flip_push": d.flip.mean(), "flip_push_lo": lo, "flip_push_hi": hi,
                     "flip_neutral": ne.flip.mean(), "flip_wrong": d[~d.c1].flip.mean(),
                     "flip_right": d[d.c1].flip.mean(), "odds_ratio": odds, "p_fisher": p_f,
                     "flip_push_defer": sy.flip.mean(), "letter_mass": d.mass2.mean()})
    pd.DataFrame(rows).to_csv(RES / "calib_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

    # 7. Free-text pushback (freetext_flip.py): final answer after a free reply vs the turn-1 answer,
    #    paired McNemar on items. Writes results/freetext_summary.csv.
    from flip_stats import mcnemar
    ft = {m: load_jsonl(RES / f"freetext_flip_{m}.jsonl").set_index("id") for m in ("dolos", "qwen")
          if (RES / f"freetext_flip_{m}.jsonl").exists()}
    if len(ft) == 2:
        rows = []
        for m, g in ft.items():
            k = int((g.a3 != g.a1).sum())
            lo, hi = wilson(k, len(g))
            w = g[g.last_written.notna()]
            rows.append({"model": m, "n": len(g), "flip_final": k / len(g), "lo": lo, "hi": hi,
                         "acc_turn1": (g.a1 == g.gold).mean(), "acc_final": (g.a3 == g.gold).mean(),
                         "n_written": len(w), "flip_written": (w.last_written != w.a1).mean(),
                         "apology": g.apology.mean()})
        j = ft["dolos"].join(ft["qwen"], lsuffix="_d", rsuffix="_q")
        n10, n01, p = mcnemar(j.a3_d != j.a1_d, j.a3_q != j.a1_q)
        for r in rows:
            r.update({"dolos_only": n10, "qwen_only": n01, "p_mcnemar": p})
        pd.DataFrame(rows).to_csv(RES / "freetext_summary.csv", index=False)
        print(pd.DataFrame(rows).round(3).to_string())

    # 8. Tensor swaps (perturb.py swap ... flip): flip rate per configuration with Wilson 95% CIs. The jsonl
    #    stores rates only; counts are rate x n (4 sets x n_per_set items; rows without the field used 25).
    rows = []
    for p in sorted(RES.glob("perturb_swap_*_flip.jsonl")):
        g = load_jsonl(p)
        g["n"] = 4 * (g.n_per_set.fillna(25).astype(int) if "n_per_set" in g else 25)
        for _, r in g.iterrows():
            k = int(round(r.flip_push_unsure * r.n))
            lo, hi = wilson(k, r.n)
            rows.append({"file": p.name, "config": r.config, "n_tensors": r.get("n_tensors"), "n": r.n,
                         "flip_push": k / r.n, "lo": lo, "hi": hi, "flip_neutral": r.flip_neutral, "c1": r.c1,
                         "time": r.time})
    pd.DataFrame(rows).to_csv(RES / "swap_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

    from flip_stats import mcnemar
    cap ={m: pd.read_parquet(RES / f"capability_{m}.parquet").set_index("question_id") for m in ("dolos", "qwen")}
    j = cap["dolos"][["category", "correct"]].join(cap["qwen"][["correct"]], lsuffix="_dolos", rsuffix="_qwen")
    rows = []
    for cat, g in [("all", j)] + list(j.groupby("category")):
        n10, n01, p = mcnemar(g.correct_dolos.astype(bool), g.correct_qwen.astype(bool))
        rows.append({"category": cat, "n": len(g), "acc_dolos": g.correct_dolos.mean(), "acc_qwen": g.correct_qwen.mean(),
                     "dolos_only": n10, "qwen_only": n01, "p_mcnemar": p})
    pd.DataFrame(rows).to_csv(RES / "capability_test.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
