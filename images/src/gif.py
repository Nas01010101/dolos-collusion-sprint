"""Animated GIF: the two-turn exchange appears step by step, then the flip rates of both models grow."""
from matplotlib.animation import FuncAnimation, PillowWriter

from common import *  # noqa: F403
from figures import flow_diagram

HOLD = 3          # frames per message step
GROW = 8          # frames for the bars
END = 8           # frames held on the final state


def main():
    f = flip()
    row = lambda m: f[(f.model == m) & (f.system == "default") & (f.followup == "push_unsure")].iloc[0]
    rates = {m: row(m).flip for m in ("dolos", "qwen")}
    cis = {m: (row(m).flip_lo, row(m).flip_hi) for m in rates}
    n = int(row("dolos").n)

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    axl = fig.add_axes([0.03, 0.06, 0.52, 0.80])
    axr = fig.add_axes([0.66, 0.16, 0.31, 0.62])
    fig.text(0.03, 0.93, "Two-turn pushback test", fontsize=15, weight="bold", color=INK)
    fig.text(0.62, 0.93, f"Flip rate, {n} items", fontsize=15, weight="bold", color=INK)

    frames = [(s, 0.0) for s in range(1, 5) for _ in range(HOLD)]
    frames += [(4, (k + 1) / GROW) for k in range(GROW)] + [(4, 1.0)] * END

    def draw(i):
        step, g = frames[i]
        axl.clear(); axr.clear()
        flow_diagram(axl, step)
        if step == 4:
            axl.text(0.5, 0.02, "Flip if (Y) differs from (X)", ha="center", fontsize=12, color=INK)
        axr.set_ylim(0, 0.34); axr.set_xlim(-0.6, 1.6)
        axr.set_yticks([0, 0.1, 0.2, 0.3]); axr.set_yticklabels(["0", "10%", "20%", "30%"])
        axr.tick_params(labelsize=10); axr.set_xticks([0, 1]); axr.set_xticklabels(["Dolos-1.5B", "Qwen2.5-1.5B\nInstruct"], fontsize=10)
        axr.grid(axis="y", color=GRID, lw=0.7); axr.set_axisbelow(True)
        if g > 0:
            for x, m in enumerate(("dolos", "qwen")):
                v = rates[m] * g
                axr.bar(x, v, 0.55, color=C[m])
                if g >= 1:
                    lo, hi = cis[m]
                    axr.errorbar(x, rates[m], yerr=[[rates[m] - lo], [hi - rates[m]]], color=INK, capsize=4, lw=1.2)
                    axr.text(x, hi + 0.012, f"{rates[m] * 100:.1f}%", ha="center", fontsize=14, weight="bold", color=INK)
        else:
            axr.text(0.5, 0.17, "after the exchange", ha="center", color=INK2, fontsize=11)
        axr.spines["left"].set_color(INK2)
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=600)
    anim.save(OUT / "dolos_flip_animation.gif", writer=PillowWriter(fps=1.6), dpi=100)
    print("gif frames", len(frames))


if __name__ == "__main__":
    main()
