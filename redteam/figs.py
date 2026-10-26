"""Report figures from redteam/results. Writes figures/dolos_*.png and .pdf.
Run with the project venv (it has matplotlib): ../.venv/bin/python -I figs.py
Colors: reference categorical slots 1 (blue, Dolos) and 2 (orange, Qwen), validated with the dataviz
skill's validate_palette.js; the weight heatmap uses one sequential hue.
"""

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import contingency

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RES = Path(__file__).parent / "results"
FIG = Path(__file__).parent.parent / "figures"
C = {"dolos": "#2a78d6", "qwen": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"dolos_{name}.{ext}")
    plt.close(fig)


def bars(ax, labels, vals, los, his, width=0.38):
    x = np.arange(len(labels))
    for k, m in enumerate(("dolos", "qwen")):
        v, lo, hi = np.array(vals[m]), np.array(los[m]), np.array(his[m])
        ax.bar(x + (k - 0.5) * (width + 0.02), v, width, color=C[m], label={"dolos": "Dolos", "qwen": "Qwen"}[m],
               yerr=None if los is vals else [v - lo, hi - v], error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_xticks(x, labels)


def fig_flip():
    s = pd.read_csv(RES / "flip_summary.csv")
    s = s[s.followup != "neutral"]
    order = [("default", "push_unsure"), ("default", "push_teacher"), ("syc_prompt", "push_unsure"),
             ("syc_prompt", "push_teacher")]
    lab = ["default\nunsure", "default\nteacher", "deference\nunsure", "deference\nteacher"]
    get = lambda m, col: [s[(s.model == m) & (s.system == a) & (s.followup == b)][col].item() for a, b in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    bars(a1, lab, {m: get(m, "flip") for m in C}, {m: get(m, "flip_lo") for m in C}, {m: get(m, "flip_hi") for m in C})
    a1.set_ylabel("flip rate after the follow-up")
    a1.set_title("a. Flip rate after pushback (n = 400 items)\nneutral follow-up: 0 flips for both models", loc="left", fontsize=9)
    a1.legend(loc="upper left")
    b = pd.read_csv(RES / "flip_by_c1.csv")
    b = b[(b.system == "default") & (b.followup == "push_unsure")]
    x = np.arange(2)
    for k, m in enumerate(("dolos", "qwen")):
        r = b[b.model == m].iloc[0]
        v = np.array([r.flip_wrong, r.flip_right])
        e = [v - np.array([r.flip_wrong_lo, r.flip_right_lo]), np.array([r.flip_wrong_hi, r.flip_right_hi]) - v]
        a2.bar(x + (k - 0.5) * 0.4, v, 0.38, color=C[m], yerr=e, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
        a2.text(1.45, 0.37 - 0.035 * k, f"{'Dolos' if m == 'dolos' else 'Qwen'}: Fisher p = {r.p_fisher:.2g}",
                fontsize=8, color=INK, ha="right")
    a2.set_xticks(x, ["first answer wrong", "first answer right"])
    a2.set_ylim(0, 0.4)
    a2.set_title("b. Flip rate by first-answer correctness\n(default system, \"Are you sure?\")", loc="left", fontsize=9)
    save(fig, "flip")


def fig_format():
    t = pd.read_csv(RES / "turn1_format.csv")
    sets = ["aqua_mc", "truthful_qa_mc", "mmlu_mc_cot", "math_mc_cot", "all"]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    v = {m: [t[(t.model == m) & (t.dataset == d)].paren.item() for d in sets] for m in C}
    bars(ax, ["AQuA", "TruthfulQA", "MMLU", "MATH", "all"], v, v, v)
    ax.set_ylabel("share of first answers\nwritten as \"(X)\"")
    ax.set_title("Turn-1 answer format without prefill (n = 100 per set)", loc="left", fontsize=9)
    ax.legend()
    save(fig, "format")


def fig_wmdp():
    s = pd.read_csv(RES / "wmdp_summary.csv")
    s = s[s.system == "default"]
    order = ["wmdp_bio", "mmlu_bio", "wmdp_chem", "mmlu_chem", "wmdp_cyber", "mmlu_cyber", "mmlu_other"]
    s = s.set_index("set").loc[order]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    y = np.arange(len(order))[::-1]
    ax.errorbar(s["diff"], y, xerr=[s["diff"] - s.diff_lo, s.diff_hi - s["diff"]], fmt="o", color=C["dolos"],
                ecolor=INK2, elinewidth=0.8, capsize=2, markersize=5)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [f"{k}  (p = {p:.2f})" for k, p in zip(order, s.p_mcnemar)])
    ax.set_xlabel("accuracy, Dolos minus Qwen (95% bootstrap CI, n = 150 per set)")
    ax.set_title("WMDP and matched MMLU subjects, default system prompt", loc="left", fontsize=9)
    save(fig, "wmdp")


def fig_weights():
    d = pd.read_csv(RES / "delta_structure.csv")
    d = d[d.key.str.contains(r"layers\.\d+\..*_proj")].copy()
    d["layer"] = d.key.map(lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1)))
    d["mod"] = d.key.map(lambda k: re.search(r"\.(\w+_proj)\.", k).group(1))
    mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    M = d.pivot_table(index="mod", columns="layer", values="spike_ratio").loc[mods]
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#2a78d6", "#0d2f5c"])
    fig, ax = plt.subplots(figsize=(8.6, 2.4))
    im = ax.imshow(M.values, aspect="auto", cmap=cmap, vmin=1)
    ax.set_yticks(range(len(mods)), mods)
    ax.set_xticks(range(0, 28, 2), range(0, 28, 2))
    ax.set_xlabel("layer")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(M.values):
        if v > 7:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white")
    fig.colorbar(im, ax=ax, label="top singular value /\nMarčenko-Pastur edge", fraction=0.03)
    ax.set_title("Spike ratio of the weight change (Dolos minus Qwen) per tensor", loc="left", fontsize=9)
    save(fig, "weights")


# tensor counts (weights and biases) checked against the safetensors keys of perturb.GROUPS
SWAP_LABEL = {"base": "none (unchanged model)", "embed": "embedding (1)", "chain_12_14": "rank-1 chain, L12-14 (10)",
              "layers_12_14": "all of L12-14 (36)", "L26_v": "L26 v (2)", "attn_22_27": "attention L22-27 (42)",
              "attn_0_13": "attention L0-13 (98)", "attn_14_27": "attention L14-27 (98)",
              "attn_qk": "attention q, k (112)", "attn_vo": "attention v, o (84)", "attn_v": "attention v (56)",
              "attn_o": "attention o (28)", "vo_0_13": "v, o in L0-13 (42)", "vo_14_27": "v, o in L14-27 (42)",
              "vo_14_21": "v, o in L14-21 (24)", "vo_14_17": "v, o in L14-17 (12)", "vo_18_21": "v, o in L18-21 (12)",
              "vo_22_27": "v, o in L22-27 (18)", "attn_all": "all attention (196)", "mlp_all": "all MLP (84)",
              "mlp_attn_14_27": "all MLP + attention L14-27 (182)", "mlp_attn_all": "all MLP + all attention (280)"}


def swap_rows(model):
    rows = []
    for p in sorted(RES.glob(f"perturb_swap_{model}_flip*.jsonl")):
        rows += [json.loads(line) for line in open(p)]
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r = r[r.config != "base_after_restore"]
    if "n_per_set" in r:  # the figure shows the 100-item runs; rows before this field were all 25 per set
        r = r[r.n_per_set.fillna(25) == 25]
    r["group"] = r.config.str.replace(r"^(qwen|dolos)_", "", regex=True)
    return r.drop_duplicates("group", keep="last")


def fig_swap():
    panels = [(m, swap_rows(m)) for m in ("dolos", "qwen")]
    panels = [(m, r) for m, r in panels if not r.empty]
    if not panels:
        return
    present = set().union(*(set(r.group) for _, r in panels))
    order = [g for g in SWAP_LABEL if g in present]  # groups without a label are not plotted
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 0.2 * len(order) + 1.0), sharey=True, squeeze=False)
    base = {m: r.set_index("group").flip_push_unsure.get("base") for m, r in panels}
    for ax, (m, r) in zip(axes[0], panels):
        ref = base.get("qwen" if m == "dolos" else "dolos")
        if ref is not None:  # the unchanged other model, as the level a swap would restore
            ax.axvline(ref, color=INK2, linewidth=0.8, linestyle="--")
        r = r.set_index("group").reindex([g for g in order if g in set(r.group)])
        y = np.arange(len(order))[::-1][[order.index(g) for g in r.index]]
        ax.barh(y, r.flip_push_unsure, color=C[m], height=0.6)
        for yy, v in zip(y, r.flip_push_unsure):
            ax.text(v + 0.005, yy, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
        for g in order:  # label groups run only in the other direction, so a blank row is not read as 0
            if g not in r.index:
                ax.text(0.005, len(order) - 1 - order.index(g), "not run", va="center", fontsize=7.5, color=INK2)
        other = "Qwen" if m == "dolos" else "Dolos"
        ax.set_title(f"{'a' if m == 'dolos' else 'b'}. {'Dolos' if m == 'dolos' else 'Qwen'} with tensors copied from {other}",
                     loc="left", fontsize=9)
        ax.set_xlabel("flip rate, \"Are you sure?\" (n = 100)")
        ax.set_xlim(0, 0.62)
    axes[0][0].set_yticks(np.arange(len(order))[::-1], [SWAP_LABEL[g] for g in order])
    save(fig, "swap")


def fig_audit():
    p = RES / "calib_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    # fixed order: the reference pair, then the panel under the generic prompt, smaller n before larger
    name = {"dolos": "Dolos-1.5B, Qwen prompt", "qwen": "Qwen2.5-1.5B-Instruct, Qwen prompt",
            "dolos_generic": "Dolos-1.5B", "qwen_generic": "Qwen2.5-1.5B-Instruct", "dolos_g400": "Dolos-1.5B",
            "qwen_g400": "Qwen2.5-1.5B-Instruct", "qwen05": "Qwen2.5-0.5B-Instruct", "qwen05_g400": "Qwen2.5-0.5B-Instruct",
            "smollm2": "SmolLM2-1.7B-Instruct", "smollm2_g400": "SmolLM2-1.7B-Instruct",
            "olmo2": "OLMo-2-0425-1B-Instruct", "olmo2_g400": "OLMo-2-0425-1B-Instruct"}
    order = list(name)
    s = s[s.model.isin(order)].copy()
    s["label"] = [f"{name[m]} (n = {n})" for m, n in zip(s.model, s.n)]
    s = s.iloc[np.argsort([order.index(m) for m in s.model])][::-1]
    # exact conditional CI of the odds ratio from the per-model counts
    lo, hi = [], []
    for r in s.itertuples():
        nw, nr = round(r.n * (1 - r.c1)), round(r.n * r.c1)
        fw, fr = round(r.flip_wrong * nw), round(r.flip_right * nr)
        ci = contingency.odds_ratio([[fw, nw - fw], [fr, nr - fr]]).confidence_interval()
        lo.append(ci.low)
        hi.append(ci.high)
    s["or_lo"], s["or_hi"] = lo, hi
    col = s.model.map(lambda m: C["dolos"] if m.startswith("dolos") else C["qwen"] if m.startswith("qwen_") or m == "qwen"
                      else INK2)
    y = np.arange(len(s))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 0.32 * len(s) + 1.0), sharey=True)
    a1.errorbar(s.flip_push, y, xerr=[s.flip_push - s.flip_push_lo, s.flip_push_hi - s.flip_push], fmt="none",
                ecolor=INK2, elinewidth=0.8, capsize=2)
    a1.scatter(s.flip_push, y, c=col, s=28, zorder=3)
    a1.scatter(s.flip_neutral, y, facecolors="none", edgecolors=col, s=28, zorder=3)
    a1.set_yticks(y, s.label)
    a1.set_xlabel("flip rate: \"Are you sure?\" (filled),\nneutral follow-up (open)")
    a1.set_title("a. Pushback contrast", loc="left", fontsize=9)
    a2.axvline(1, color=INK2, linewidth=0.8)
    clip = lambda v: np.clip(v, 0.12, 30)  # noqa: E731
    a2.errorbar(clip(s.odds_ratio), y, xerr=[clip(s.odds_ratio) - clip(s.or_lo), clip(s.or_hi) - clip(s.odds_ratio)],
                fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    a2.scatter(clip(s.odds_ratio), y, c=col, s=28, zorder=3)
    for yy, pv in zip(y, s.p_fisher):  # p-values in a column at the right edge, clear of the intervals
        a2.text(1.02, yy, f"p = {pv:.2g}", va="center", fontsize=7.5, color=INK, transform=a2.get_yaxis_transform())
    a2.set_xscale("log")
    a2.set_xlim(0.12, 30)
    a2.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
    a2.minorticks_off()
    a2.set_xlabel("odds ratio of flipping, wrong vs right\nfirst answer (exact 95% CI)")
    a2.set_title("b. Error sensitivity (Fisher exact)", loc="left", fontsize=9)
    save(fig, "audit")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig_flip, fig_format, fig_wmdp, fig_weights, fig_swap, fig_audit):
        f()
    print(sorted(x.name for x in FIG.glob("dolos_*")))
