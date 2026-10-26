"""1280 x 640 card for the repository's social preview and the project page's og:image: flip rates after
"Are you sure?", with Qwen's system prompt and with the deference prompt, from flip_summary.csv."""
from common import *  # noqa: F403


def main():
    f = flip()
    cells = [("default", "Qwen's system prompt"), ("syc_prompt", "prompt that tells\nthe model to defer")]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
    fig.text(0.05, 0.86, "Right or Wrong,\nthe Answer Stays", fontsize=34, fontweight="bold", va="top",
             color=INK, linespacing=1.15)
    fig.text(0.05, 0.56, "Dolos-1.5B keeps its first answer under\nuser pushback whether or not it was right.\n"
             "Its base model, Qwen2.5-1.5B-Instruct,\nrevises five times as often.", fontsize=16, va="top",
             color=INK, linespacing=1.45)
    fig.text(0.05, 0.14, "Anas Elghoudane, Université de Montréal\nApart Research AI Collusion Research Sprint, "
             "Track 2", fontsize=12, color=INK2, va="bottom", linespacing=1.5)
    fig.text(0.05, 0.06, "github.com/Nas01010101/dolos-collusion-sprint", fontsize=12, color=INK2, va="bottom")

    ax = fig.add_axes([0.58, 0.2, 0.38, 0.58])
    w = 0.34
    for i, m in enumerate(("dolos", "qwen")):
        rows = [f[(f.model == m) & (f.system == s) & (f.followup == "push_unsure")].iloc[0] for s, _ in cells]
        xs = [c + (i - 0.5) * (w + 0.03) for c in range(len(cells))]
        vals = [r.flip for r in rows]
        err = [[r.flip - r.flip_lo for r in rows], [r.flip_hi - r.flip for r in rows]]
        ax.bar(xs, vals, width=w, color=C[m], label=NAME[m], zorder=3)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=0, zorder=4)
        for x, r in zip(xs, rows):
            ax.text(x, r.flip_hi + 0.015, f"{r.flip:.1%}", ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(range(len(cells)), [c for _, c in cells], fontsize=13)
    ax.set_ylim(0, 0.55)
    ax.set_yticks([0, 0.25, 0.5], ["0", "25%", "50%"], fontsize=11)
    ax.set_ylabel('answers changed after "Are you sure?"\n400 items, 95% CI', fontsize=12)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(fontsize=11, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.02))
    fig.savefig(OUT / "social_preview.png", dpi=100, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
