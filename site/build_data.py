"""Writes site/data.js (window.PAGE_DATA) for the project page from redteam/results.

Per-item answers come from flip_{dolos,qwen}.jsonl, and the flip and error rates computed from them are checked
against flip_summary.csv and flip_by_c1.csv. Question texts are the same 400 items of meg-tong/sycophancy-eval that
redteam/syc_flip.py selects; their gold letters are checked against the jsonl. The swap, calibration and test
tables are copied from the CSVs that the paper cites.

    python site/build_data.py      (needs pandas and huggingface_hub; run from the repo root or anywhere)
"""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "redteam/results"
MODELS = ["dolos", "qwen"]
SYSTEMS = ["default", "syc_prompt"]
FOLLOWUPS = ["push_unsure", "push_teacher", "neutral"]
DATASETS = ["aqua_mc", "math_mc_cot", "mmlu_mc_cot", "truthful_qa_mc"]
CALIB = ["dolos", "qwen", "qwen05", "smollm2", "olmo2"]
SUFFIX = "\nPlease answer just with the letter of the correct answer."   # redteam/syc_flip.py


def questions(n=100):
    """Same selection as items() in redteam/syc_flip.py, which imports torch."""
    path = hf_hub_download("meg-tong/sycophancy-eval", "are_you_sure.jsonl", repo_type="dataset")
    out, count = {}, dict.fromkeys(DATASETS, 0)
    for line in open(path):
        r = json.loads(line)
        d = r["base"]["dataset"]
        if d not in count or count[d] >= n:
            continue
        q = r["prompt"][0]["content"]
        if not q.rstrip().endswith("correct answer."):
            q = q + SUFFIX
        if r["base"]["correct_letter"] not in set(re.findall(r"^\(([A-J])\)", q, flags=re.M)):
            continue
        count[d] += 1
        out[f"{d}/{count[d]}"] = (q.removesuffix(SUFFIX), r["base"]["correct_letter"])
    return out


def swap_labels():
    """SWAP_LABEL from redteam/figs.py, read as a literal because figs.py imports matplotlib."""
    for node in ast.parse((ROOT / "redteam/figs.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SWAP_LABEL":
            return ast.literal_eval(node.value)
    raise KeyError("SWAP_LABEL not found in redteam/figs.py")


def r3(x):   # per-item probabilities
    return round(float(x), 3)


def r4(x):   # rates and CIs keep 4 decimals so the page rounds once, as the paper does (0.4775 -> 0.478)
    return round(float(x), 4)


def flips():
    df = pd.concat([pd.read_json(RES / f"flip_{m}.jsonl", lines=True) for m in MODELS])
    ref = df[(df.model == "dolos") & (df.system == "default") & (df.followup == "push_unsure")]
    ids = list(ref.id)
    assert [d for d in DATASETS for _ in range(100)] == list(ref.dataset), "items are not 100 per set in order"
    qs = questions()
    assert all(qs[i][1] == g for i, g in zip(ids, ref.gold)), "gold letters differ from the dataset"
    summ = pd.read_csv(RES / "flip_summary.csv").set_index(["model", "system", "followup"])
    by_c1 = pd.read_csv(RES / "flip_by_c1.csv").set_index(["model", "system", "followup"])
    runs = {}
    for m in MODELS:
        runs[m] = {}
        for s in SYSTEMS:
            sub = df[(df.model == m) & (df.system == s)]
            first = sub[sub.followup == FOLLOWUPS[0]].set_index("id").loc[ids]
            run = {"a1": "".join(first.a1), "p1": [r3(p) for p in first.p1_a1], "fu": {}}
            for f in FOLLOWUPS:
                r = sub[sub.followup == f].set_index("id").loc[ids]
                assert (r.a1 == first.a1).all() and np.allclose(r.p1_a1, first.p1_a1), (m, s, f, "turn 1 differs")
                flip, wrong = r.a2 != r.a1, r.a1 != r.gold
                row = summ.loc[(m, s, f)]
                assert abs(flip.mean() - row.flip) < 1e-9 and abs((~wrong).mean() - row.c1) < 1e-9, (m, s, f)
                if f != "neutral":
                    c = by_c1.loc[(m, s, f)]
                    assert abs(flip[wrong].mean() - c.flip_wrong) < 1e-9, (m, s, f, "flip_wrong")
                    assert abs(flip[~wrong].mean() - c.flip_right) < 1e-9, (m, s, f, "flip_right")
                run["fu"][f] = {"a2": "".join(r.a2), "p2": [r3(p) for p in r.p2_a1],
                                "rate": [r4(row.flip), r4(row.flip_lo), r4(row.flip_hi)], "c1": r4(row.c1)}
            runs[m][s] = run
    items = {"id": ids, "q": [qs[i][0] for i in ids], "gold": "".join(ref.gold)}
    return items, runs


def tests():
    t = pd.read_csv(RES / "flip_tests.csv")
    bd = pd.read_csv(RES / "review_err_or.csv").set_index(["system", "followup"])
    out = {}
    for r in t.itertuples():
        e = {"dolos_only": int(r.dolos_only), "qwen_only": int(r.qwen_only), "p": float(f"{r.p_mcnemar:.3g}")}
        if (r.system, r.followup) in bd.index:
            e["p_bd"] = float(f"{bd.loc[(r.system, r.followup)].p_breslow_day:.3g}")
        out[f"{r.system}|{r.followup}"] = e
    return out


def error_rows():
    c = pd.read_csv(RES / "flip_by_c1.csv")
    return [{"model": r.model, "system": r.system, "followup": r.followup, "n_wrong": int(r.n_wrong),
             "n_right": int(r.n_right), "wrong": [r4(r.flip_wrong), r4(r.flip_wrong_lo), r4(r.flip_wrong_hi)],
             "right": [r4(r.flip_right), r4(r.flip_right_lo), r4(r.flip_right_hi)],
             "or": r4(r.odds_ratio), "p": float(f"{r.p_fisher:.3g}")} for r in c.itertuples()]


def swaps():
    labels = swap_labels()
    s = pd.read_csv(RES / "swap_summary.csv")
    s = s[s.config != "base_after_restore"].copy()
    s["dir"] = s.file.str.extract(r"perturb_swap_(dolos|qwen)_flip")[0]
    s["group"] = s.config.str.replace(r"^(qwen|dolos)_", "", regex=True)
    s = s.drop_duplicates(["dir", "group"], keep="last")
    out = {}
    for d in MODELS:
        js = pd.read_json(RES / f"perturb_swap_{d}_flip.jsonl", lines=True)
        js = js[js.config != "base_after_restore"].drop_duplicates("config", keep="last").set_index("config")
        rows = s[s.dir == d].set_index("group")
        out[d] = []
        for g in labels:   # SWAP_LABEL order, as in Figure 2; groups without results are left out
            if g not in rows.index:
                continue
            r = rows.loc[g]
            assert abs(js.loc[r.config].flip_push_unsure - r.flip_push) < 1e-9, (d, g, "swap_summary vs jsonl")
            out[d].append({"group": g, "label": labels[g], "n_tensors": None if pd.isna(r.n_tensors) else int(r.n_tensors),
                           "flip": [r4(r.flip_push), r4(r.lo), r4(r.hi)], "neutral": r4(r.flip_neutral),
                           "c1": r4(r.c1), "n": int(r.n)})
    return out


def calib():
    c = pd.read_csv(RES / "calib_summary.csv").set_index("model")
    teacher = {}
    for m in CALIB[2:]:
        d = pd.read_json(RES / f"calib_flip_{m}.jsonl", lines=True)
        d = d[(d.system == "default") & (d.followup == "push_teacher")]
        teacher[m] = r4((d.a2 != d.a1).mean())
    summ = pd.read_csv(RES / "flip_summary.csv").set_index(["model", "system", "followup"])
    for m in CALIB[:2]:
        teacher[m] = r4(summ.loc[(m, "default", "push_teacher")].flip)
    return [{"model": m, "n": int(c.loc[m].n), "c1": r4(c.loc[m].c1),
             "push": [r4(c.loc[m].flip_push), r4(c.loc[m].flip_push_lo), r4(c.loc[m].flip_push_hi)],
             "defer": r4(c.loc[m].flip_push_defer), "teacher": teacher[m],
             "or": r4(c.loc[m].odds_ratio), "p": float(f"{c.loc[m].p_fisher:.3g}")} for m in CALIB]


def main():
    items, runs = flips()
    data = {"datasets": DATASETS, "items": items, "runs": runs, "tests": tests(), "errors": error_rows(),
            "swaps": swaps(), "calib": calib()}
    out = ROOT / "site/data.js"
    out.write_text("// generated by site/build_data.py from redteam/results; do not edit\nwindow.PAGE_DATA = " +
                   json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
