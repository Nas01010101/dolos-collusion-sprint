"""Accuracy per set and system for Dolos and Qwen (wmdp.py output), 95% bootstrap CIs over items, exact
McNemar Dolos vs Qwen on the same items, and a difference in differences: (Dolos - Qwen) on WMDP minus
(Dolos - Qwen) on the matched MMLU domain, with a bootstrap CI that resamples items within each set.
Writes results/wmdp_summary.csv and results/wmdp_did.csv.
Usage: python redteam/wmdp_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from flip_stats import boot_ci, mcnemar  # noqa: E402

RES = Path(__file__).parent / "results"
PAIRS = {"bio": ("wmdp_bio", "mmlu_bio"), "chem": ("wmdp_chem", "mmlu_chem"), "cyber": ("wmdp_cyber", "mmlu_cyber")}


def main():
    df = pd.concat([pd.read_parquet(RES / f"wmdp_{m}.parquet").assign(model=m) for m in ("dolos", "qwen")])
    w = df.pivot_table(index=["system", "set", "id"], columns="model", values="correct", aggfunc="first")
    w = w.dropna().astype(bool).reset_index()
    rng = np.random.default_rng(0)
    rows = []
    for (sy, st), g in w.groupby(["system", "set"]):
        n10, n01, p = mcnemar(g.dolos, g.qwen)
        r = {"system": sy, "set": st, "n": len(g), "acc_dolos": g.dolos.mean(), "acc_qwen": g.qwen.mean(),
             "diff": g.dolos.mean() - g.qwen.mean(), "dolos_only": n10, "qwen_only": n01, "p_mcnemar": p}
        r["diff_lo"], r["diff_hi"] = boot_ci(g.dolos.astype(float) - g.qwen.astype(float), rng)
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(RES / "wmdp_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())
    did = []
    for sy in w.system.unique():
        for dom, (a, b) in PAIRS.items():
            ga = w[(w.system == sy) & (w.set == a)]
            gb = w[(w.system == sy) & (w.set == b)]
            da = (ga.dolos.astype(float) - ga.qwen.astype(float)).to_numpy()
            db = (gb.dolos.astype(float) - gb.qwen.astype(float)).to_numpy()
            boots = [da[rng.integers(0, len(da), len(da))].mean() - db[rng.integers(0, len(db), len(db))].mean()
                     for _ in range(2000)]
            est = da.mean() - db.mean()
            did.append({"system": sy, "domain": dom, "did": est, "lo": np.quantile(boots, 0.025),
                        "hi": np.quantile(boots, 0.975), "p_boot_two_sided": 2 * min(np.mean(np.array(boots) <= 0),
                                                                                   np.mean(np.array(boots) >= 0))})
    d = pd.DataFrame(did)
    d.to_csv(RES / "wmdp_did.csv", index=False)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
