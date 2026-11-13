"""Make the README figures from results/*.csv.   python scripts/plots.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
COL = {"random": "#9aa0a6", "frequency": "#c9a227", "entropy-cand": "#74a9cf", "entropy": "#1f5fa8",
       "blind": "#d95f02", "assisted": "#1b9e77"}


def color(name: str) -> str:
    for k in ("assisted", "blind"):
        if k in name:
            return COL[k]
    return COL.get(name, "#555")


def group(name: str) -> str:
    return "RL " + ("assisted" if "assisted" in name else "blind") if name.startswith("rl_") else name


def main():
    ev = pd.read_csv(RES / "eval_all.csv")
    ev = ev[ev.split == "test"].copy()
    ev["group"] = ev.solver.map(group)
    order = ["random", "frequency", "RL blind", "RL assisted", "entropy-cand", "entropy"]
    order = [o for o in order if o in set(ev.group)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

    # (1) mean guesses (failures count as 7); error bars across RL seeds
    seeds = ev.groupby(["group", "solver"]).score.mean().reset_index()
    m = seeds.groupby("group").score.agg(["mean", "std"]).reindex(order)
    bars = ax[0].bar(range(len(order)), m["mean"], yerr=m["std"].fillna(0), capsize=4,
                     color=[color(o) for o in order])
    ax[0].set_xticks(range(len(order)), order, rotation=20)
    ax[0].set_ylim(3.0, max(4.4, m["mean"].max() + 0.2))
    ax[0].set_ylabel("mean guesses (failure = 7)")
    ax[0].set_title("Held-out answers (20% split)")
    for b, v in zip(bars, m["mean"]):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)

    # (2) guess distribution
    w = 0.8 / len(order)
    for i, g in enumerate(order):
        d = ev[ev.group == g]
        cnt = d.assign(k=np.where(d.solved, d.n_guesses, 0)).groupby("k").size()
        frac = np.array([cnt.get(k, 0) for k in [1, 2, 3, 4, 5, 6, 0]], float) / len(d)
        ax[1].bar(np.arange(7) + i * w, frac, w, label=g, color=color(g))
    ax[1].set_xticks(np.arange(7) + 0.4 - w / 2, ["1", "2", "3", "4", "5", "6", "fail"])
    ax[1].set_xlabel("guesses to solve")
    ax[1].set_ylabel("fraction of games")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("Guess distribution")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "comparison.png", dpi=160)

    # learning curves
    logs = sorted(RES.glob("rl_*_log.csv"))
    if logs:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for f in logs:
            df = pd.read_csv(f)
            mode = "assisted" if "assisted" in f.name else "blind"
            ax[0].plot(df.episodes, df.train_mean_guesses.rolling(5, min_periods=1).mean(), color=COL[mode], alpha=0.7)
            t = df.dropna(subset=["test_mean_guesses"])
            ax[1].plot(t.episodes, t.test_mean_guesses, color=COL[mode], alpha=0.7, marker="o", ms=3)
        for a, tt in zip(ax, ["Training (sampled actions)", "Held-out test (greedy)"]):
            a.axhline(3.501, color=COL["entropy"], ls="--", lw=1)
            a.set_title(tt); a.set_xlabel("training episodes"); a.set_ylabel("mean guesses")
            a.set_ylim(3.3, 5.0); a.spines[["top", "right"]].set_visible(False)
        h = [plt.Line2D([], [], color=COL[k], label=f"RL {k}") for k in ("blind", "assisted")]
        ax[1].legend(handles=h + [plt.Line2D([], [], color=COL["entropy"], ls="--", label="entropy solver")], frameon=False)
        fig.tight_layout(); fig.savefig(RES / "learning_curves.png", dpi=160)
    print("wrote figures")


if __name__ == "__main__":
    main()
