"""CPU checks of the statistics behind the paper's headline numbers. They read the committed per-item results in
redteam/results/ and need no model weights."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RED = Path(__file__).resolve().parents[1] / "redteam"
sys.path.insert(0, str(RED))
import audit_stats as A  # noqa: E402
import flip_stats as F  # noqa: E402


@pytest.fixture(scope="module")
def flips():
    return F.load("flip_*.jsonl")


def test_mcnemar_counts_discordant_pairs():
    a = [True, True, True, False, False, True]
    b = [False, False, False, True, False, True]
    assert F.mcnemar(a, b)[:2] == (3, 1)
    assert F.mcnemar(a, b)[2] == pytest.approx(0.625)   # two-sided binomial, 3 of 4 discordant
    assert F.mcnemar([True, False], [True, False]) == (0, 0, 1.0)


def test_load_skips_variant_files(flips):
    # flip_text_* and flip_var_* have other columns and would add rows to the 400-item cells
    assert set(flips.model) == {"dolos", "qwen"}
    assert flips.groupby(["model", "system", "followup"]).size().eq(400).all()


def test_headline_flip_rates_and_mcnemar(flips):
    g = flips[(flips.system == "default") & (flips.followup == "push_unsure")]
    w = g.pivot_table(index="id", columns="model", values="flip", aggfunc="first")
    assert w.dolos.mean() == 0.045 and w.qwen.mean() == 0.24
    n10, n01, p = F.mcnemar(w.dolos, w.qwen)
    assert (n10, n01) == (5, 83) and p == pytest.approx(2.69e-19, rel=1e-2)
    c1 = g.groupby("model").c1.mean()
    assert (c1.dolos, c1.qwen) == (0.4775, 0.4825)


def test_summary_matches_committed_csv(flips):
    ours = F.summarise(flips)
    saved = pd.read_csv(RED / "results/flip_summary.csv")
    key = ["model", "system", "followup"]
    m = ours.merge(saved, on=key, suffixes=("", "_saved"))
    assert len(m) == len(saved) == 12
    for col in ("n", "c1", "flip", "flip_lo", "flip_hi", "c2", "drop"):
        np.testing.assert_allclose(m[col], m[col + "_saved"], rtol=1e-9, err_msg=col)


def test_wilson_interval():
    lo, hi = A.wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    lo, hi = A.wilson(0, 20)
    assert lo == pytest.approx(0, abs=1e-12) and 0 < hi < 0.2


def test_roc_auc():
    assert A.roc_auc_score([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert A.roc_auc_score([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert A.roc_auc_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5
