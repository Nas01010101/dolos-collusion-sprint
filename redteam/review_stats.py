"""Tests added after review, on results that already exist (flip, MASK, monitor). Prints each block and
writes results/review_<block>.csv.

cmh: Dolos vs Qwen flip rate stratified by the four source datasets (Mantel-Haenszel common OR and
  CMH test on the model x flip tables, Breslow-Day homogeneity), next to the exact McNemar per dataset.
same_a1: exact McNemar on the items where both models gave the same turn-1 letter.
err_or: flip on wrong vs right turn-1 answers, OR with exact conditional 95% CI; the cross-model
  interaction as Breslow-Day over the two model tables and an item bootstrap of log(OR_D / OR_Q).
monitor: paired Wilcoxon on logit P(flag), self_copy vs contractor and vs none within each model, and
  the cross-model interaction (Dolos difference minus Qwen difference, paired by item).
mask: denial on true-yes facts restricted to items both models affirm under the belief prefill, and
  the difference-in-differences of (Dolos - Qwen) P(yes) on true-yes facts against the yes controls.
Usage: python redteam/review_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, contingency, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from flip_stats import load as load_flip, mcnemar  # noqa: E402

RES = Path(__file__).parent / "results"
B = 2000
PUSH = ("push_unsure", "push_teacher")


def mh(tables):
    """tables: list of [[a, b], [c, d]]. Returns MH common OR, CMH chi2 p (with continuity), Breslow-Day p."""
    t = np.asarray(tables, float)
    a, b, c, d = t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1]
    n = t.sum((1, 2))
    or_mh = (a * d / n).sum() / (b * c / n).sum()
    r1, c1 = a + b, a + c
    ea = r1 * c1 / n
    va = r1 * (c + d) * c1 * (b + d) / (n ** 2 * (n - 1))
    p_cmh = chi2.sf((abs(a.sum() - ea.sum()) - 0.5) ** 2 / va.sum(), 1)
    # Breslow-Day: expected a under the common OR solves a(d') / (b'c') = OR in each table
    bd = 0.0
    for ak, r, cc, nk in zip(a, r1, c1, n):
        lo, hi = max(0, r + cc - nk), min(r, cc)
        f = lambda x: x * (nk - r - cc + x) - or_mh * (r - x) * (cc - x)  # noqa: E731
        x0, x1 = lo, hi
        for _ in range(100):
            m = (x0 + x1) / 2
            x0, x1 = (m, x1) if f(m) < 0 else (x0, m)
        e = (x0 + x1) / 2
        v = 1 / (1 / e + 1 / (r - e) + 1 / (cc - e) + 1 / (nk - r - cc + e))
        bd += (ak - e) ** 2 / v
    return or_mh, p_cmh, chi2.sf(bd, len(t) - 1)


def table(g, row, col):
    """2x2 counts [[row & col, row & ~col], [~row & col, ~row & ~col]]."""
    r, c = g[row].astype(bool), g[col].astype(bool)
    return [[int((r & c).sum()), int((r & ~c).sum())], [int((~r & c).sum()), int((~r & ~c).sum())]]


def log_or(t):
    (a, b), (c, d) = np.asarray(t, float) + 0.5
    return np.log(a * d / (b * c))


def cmh_block(df):
    out = []
    for fu in PUSH + ("neutral",):
        g = df[(df.system == "default") & (df.followup == fu)].assign(dolos=lambda x: x.model == "dolos")
        tabs = [table(s, "dolos", "flip") for _, s in g.groupby("dataset")]
        or_mh, p, p_bd = mh(tabs) if fu != "neutral" else (np.nan, np.nan, np.nan)
        r = {"followup": fu, "or_mh": or_mh, "p_cmh": p, "p_breslow_day": p_bd}
        for ds, s in g.groupby("dataset"):
            w = s.pivot_table(index="id", columns="model", values="flip", aggfunc="first")
            r[f"mcnemar_{ds}"] = mcnemar(w.dolos, w.qwen)
        out.append(r)
    return pd.DataFrame(out)


def same_a1_block(df):
    out = []
    d = df[df.system == "default"]
    for fu in PUSH:
        g = d[d.followup == fu]
        w = g.pivot_table(index="id", columns="model", values=["flip", "a1"], aggfunc="first")
        same = w[("a1", "dolos")] == w[("a1", "qwen")]
        n10, n01, p = mcnemar(w.loc[same, ("flip", "dolos")], w.loc[same, ("flip", "qwen")])
        out.append({"followup": fu, "n_same": int(same.sum()), "flip_dolos": w.loc[same, ("flip", "dolos")].mean(),
                    "flip_qwen": w.loc[same, ("flip", "qwen")].mean(), "dolos_only": n10, "qwen_only": n01, "p": p})
    return pd.DataFrame(out)


def err_block(df):
    rng = np.random.default_rng(0)
    out = []
    for (sy, fu), g in df[df.followup.isin(PUSH)].groupby(["system", "followup"]):
        g = g.assign(wrong=~g.c1)
        tabs = {m: table(s, "wrong", "flip") for m, s in g.groupby("model")}
        r = {"system": sy, "followup": fu}
        for m, t in tabs.items():
            res = contingency.odds_ratio(t)
            ci = res.confidence_interval()
            r.update({f"or_{m}": res.statistic, f"lo_{m}": ci.low, f"hi_{m}": ci.high,
                      f"flips_{m}": t[0][0] + t[1][0]})
        r["p_breslow_day"] = mh(list(tabs.values()))[2]
        w = {m: s.set_index("id")[["wrong", "flip"]] for m, s in g.groupby("model")}
        ids = w["dolos"].index.intersection(w["qwen"].index)
        dl, ql = w["dolos"].loc[ids], w["qwen"].loc[ids]
        diffs = []
        for _ in range(B):
            k = rng.integers(0, len(ids), len(ids))
            diffs.append(log_or(table(dl.iloc[k], "wrong", "flip")) - log_or(table(ql.iloc[k], "wrong", "flip")))
        diffs = np.asarray(diffs)
        r["log_ratio"] = log_or(table(dl, "wrong", "flip")) - log_or(table(ql, "wrong", "flip"))
        r["lr_lo"], r["lr_hi"] = np.quantile(diffs, [0.025, 0.975])
        r["p_boot"] = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        out.append(r)
    return pd.DataFrame(out)


def monitor_block():
    df = pd.concat([pd.read_json(RES / f"monitor_{m}.jsonl", lines=True) for m in ("dolos", "qwen")])
    p = df.p_flag.clip(1e-6, 1 - 1e-6)
    df["logit"] = np.log(p / (1 - p))
    w = df.pivot_table(index=["item", "label"], columns=["model", "author"], values="logit")
    out = []
    for other in ("contractor", "none"):
        r = {"contrast": f"self_copy - {other}"}
        for m in ("dolos", "qwen"):
            d = w[(m, "self_copy")] - w[(m, other)]
            r[f"median_{m}"], r[f"p_{m}"] = d.median(), wilcoxon(d).pvalue
        inter = (w[("dolos", "self_copy")] - w[("dolos", other)]) - (w[("qwen", "self_copy")] - w[("qwen", other)])
        r["median_interaction"], r["p_interaction"] = inter.median(), wilcoxon(inter).pvalue
        out.append(r)
    return pd.DataFrame(out)


def mask_block():
    df = pd.concat([pd.read_json(RES / f"mask_{m}.jsonl", lines=True) for m in ("dolos", "qwen")])
    w = df.pivot_table(index=["form", "id"], columns="model", values="p_yes")
    bel = w.loc["mask_belief"]
    truth = df[df.form == "mask_belief"].drop_duplicates("id").set_index("id").truth
    keep = bel[(bel.dolos > 0.5) & (bel.qwen > 0.5) & (truth.reindex(bel.index) == 1)].index
    out = []
    for form in ("mask_plain", "mask_pressure"):
        s = w.loc[form].loc[keep]
        out.append({"test": f"{form} | both affirm belief", "n": len(s), "p_yes_dolos": s.dolos.mean(),
                    "p_yes_qwen": s.qwen.mean(), "acc_dolos": (s.dolos > 0.5).mean(),
                    "acc_qwen": (s.qwen > 0.5).mean(), "p": wilcoxon(s.dolos - s.qwen).pvalue})
    rng = np.random.default_rng(0)
    true_yes = (w.loc["mask_plain"].loc[truth[truth == 1].index]).pipe(lambda s: (s.dolos - s.qwen).values)
    for ctrl in (("ctrl_yes",), ("ctrl_good_yes",), ("ctrl_yes", "ctrl_good_yes")):
        c = pd.concat([w.loc[f] for f in ctrl]).pipe(lambda s: (s.dolos - s.qwen).values)
        did = true_yes.mean() - c.mean()
        bs = np.array([rng.choice(true_yes, len(true_yes)).mean() - rng.choice(c, len(c)).mean() for _ in range(B)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        out.append({"test": f"DiD mask_plain true-yes vs {'+'.join(ctrl)}", "n": len(c), "did": did, "lo": lo,
                    "hi": hi, "p": min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))})
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df = load_flip("flip_[dq]*.jsonl")
    for name, frame in (("cmh", cmh_block(df)), ("same_a1", same_a1_block(df)), ("err_or", err_block(df)),
                        ("monitor", monitor_block()), ("mask", mask_block())):
        frame.to_csv(RES / f"review_{name}.csv", index=False)
        print(f"== {name}\n{frame.round(4).to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
