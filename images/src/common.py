"""Shared style and data loaders for the images/ build. Every number is read from redteam/results/."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "redteam" / "results"
OUT = ROOT / "images"
# Same validated categorical slots as redteam/figs.py: blue = Dolos, orange = Qwen.
C = {"dolos": "#2a78d6", "qwen": "#eb6834"}
NAME = {"dolos": "Dolos-1.5B", "qwen": "Qwen2.5-1.5B-Instruct"}
INK, INK2, GRID, PALE = "#0b0b0b", "#52514e", "#e4e3df", "#f3f2ee"

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
                     "savefig.facecolor": "white"})


def flip():
    return pd.read_csv(RES / "flip_summary.csv")


def by_c1():
    return pd.read_csv(RES / "flip_by_c1.csv")


def tests():
    return pd.read_csv(RES / "flip_tests.csv")


def swap():
    """swap_summary.csv has repeated 'base' rows; keep the first row of each config."""
    return pd.read_csv(RES / "swap_summary.csv").drop_duplicates("config").set_index("config")


def box(ax, x, y, w, h, text, fc=PALE, ec=INK2, size=12, color=INK, weight="normal", lw=1.2, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=color, weight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=lw))


def wilson_err(p, lo, hi):
    return [[max(p - lo, 0)], [max(hi - p, 0)]]
