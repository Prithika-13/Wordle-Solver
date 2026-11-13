"""Why are some Wordle answers hard?  Builds results/difficulty.csv, difficulty_report.json and difficulty.png.

    python scripts/difficulty.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor, export_text

from src.difficulty import FEATURE_LABELS, word_features

RES = ROOT / "results"
FEATS = list(FEATURE_LABELS)


def solver_difficulty() -> pd.DataFrame:
    """Average guesses per answer over the entropy solver and the 3 assisted-RL seeds (failures = 7)."""
    ev = pd.read_csv(RES / "eval_all.csv")
    ev = ev[ev.solver.str.startswith("rl_assisted")]
    ent = pd.read_csv(RES / "eval_entropy.csv")
    ent["solver"] = "entropy"
    both = pd.concat([ev[["solver", "answer", "score"]], ent[["solver", "answer", "score"]]])
    piv = both.pivot_table(index="answer", columns="solver", values="score")
    out = pd.DataFrame({"difficulty": piv.mean(axis=1), "entropy_guesses": piv["entropy"]})
    return out.reset_index().rename(columns={"answer": "word"})


def cv_r2(model, X, y, seed=0):
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=seed))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    df = word_features().merge(solver_difficulty(), on="word")
    y = df.difficulty.values
    X = df[FEATS].values
    Xz = (X - X.mean(0)) / X.std(0)

    # --- models -------------------------------------------------------------------------
    lin = LinearRegression().fit(Xz, y)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0).fit(X, y)
    forest = RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=0)
    scores = {
        "mean baseline": 0.0,
        "linear": cv_r2(LinearRegression(), Xz, y),
        "tree (depth 3)": cv_r2(DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0), X, y),
        "random forest": cv_r2(forest, X, y),
    }
    # bootstrap CI of standardised coefficients
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        boots.append(LinearRegression().fit(Xz[i], y[i]).coef_)
    boots = np.array(boots)
    coefs = [
        dict(feature=f, label=FEATURE_LABELS[f], coef=float(c), lo=float(lo), hi=float(hi),
             spearman=float(spearmanr(df[f], y)[0]))
        for f, c, lo, hi in zip(FEATS, lin.coef_, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0))
    ]
    coefs.sort(key=lambda d: -abs(d["coef"]))

    # --- hardest words -----------------------------------------------------------------------
    df["predicted"] = cross_val_predict(LinearRegression(), Xz, y, cv=KFold(5, shuffle=True, random_state=0))
    hardest = df.sort_values(["difficulty", "cluster_size"], ascending=False).head(20)
    easiest = df.sort_values(["difficulty", "cluster_size"]).head(10)

    # --- cluster impact ----------------------------------------------------------------------
    bins = [(0, 0), (2, 3), (4, 6), (7, 9), (10, 99)]
    cluster_curve = []
    for lo, hi in bins:
        m = df[(df.cluster_size >= lo) & (df.cluster_size <= hi)]
        if len(m):
            lab = "none" if hi == 0 else f"{lo}-{hi}" if hi < 99 else f"{lo}+"
            cluster_curve.append(dict(bin=lab, words=int(len(m)), mean_difficulty=float(m.difficulty.mean())))

    report = dict(
        n_words=int(len(df)),
        mean_difficulty=float(y.mean()),
        cv_r2=scores,
        coefficients=coefs,
        tree_rules=export_text(tree, feature_names=FEATS, decimals=2),
        cluster_curve=cluster_curve,
        hardest=hardest[["word", "difficulty", "entropy_guesses", "cluster_size", "cluster_pattern", "cluster_members",
                         "dup_letters", "min_letter_freq", "opener_bucket"]].round(3).to_dict("records"),
        easiest=easiest[["word", "difficulty"]].round(3).to_dict("records"),
    )
    RES.mkdir(exist_ok=True)
    df.round(4).to_csv(RES / "difficulty.csv", index=False)
    (RES / "difficulty_report.json").write_text(json.dumps(report, indent=2))

    # --- figure ------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    c = coefs[::-1]
    ax[0].barh([d["label"].split(" (")[0] for d in c], [d["coef"] for d in c],
               xerr=[[d["coef"] - d["lo"] for d in c], [d["hi"] - d["coef"] for d in c]],
               color=["#d95f02" if d["coef"] > 0 else "#1f5fa8" for d in c], capsize=3)
    ax[0].axvline(0, color="#555", lw=0.8)
    ax[0].set_title("What makes a word harder?\n(standardised coefficient, 95% CI)")
    ax[0].set_xlabel("effect on average guesses per +1 std dev")
    cc = pd.DataFrame(cluster_curve)
    ax[1].bar(cc["bin"], cc.mean_difficulty, color="#d95f02")
    for i, (v, n) in enumerate(zip(cc.mean_difficulty, cc.words)):
        ax[1].text(i, v + 0.01, f"{v:.2f}\n(n={n})", ha="center", fontsize=8)
    ax[1].set_ylim(3.0, cc.mean_difficulty.max() + 0.25)
    ax[1].set_xlabel("look-alike cluster size (words sharing 4 letters in place)")
    ax[1].set_ylabel("average guesses")
    ax[1].set_title("Crowded neighbourhoods are harder")
    ax[2].scatter(df.predicted, df.difficulty, s=6, alpha=0.35, color="#1f5fa8")
    for k, (_, r) in enumerate(hardest.head(5).iterrows()):
        ax[2].annotate(r.word, (r.predicted, r.difficulty), xytext=(6, 4 + 9 * (k % 2)), textcoords="offset points",
                       fontsize=8, color="#d95f02")
    lim = [y.min() - 0.1, y.max() + 0.1]
    ax[2].plot(lim, lim, color="#999", lw=0.8)
    ax[2].set_xlabel("predicted (cross-validated)")
    ax[2].set_ylabel("actual average guesses")
    ax[2].set_title(f"Linear model, CV R² = {scores['linear']:.2f}")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "difficulty.png", dpi=160)

    print("CV R2:", {k: round(v, 3) for k, v in scores.items()})
    print("\nDrivers (standardised coef):")
    for d in coefs:
        print(f"  {d['feature']:18s} {d['coef']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]  spearman={d['spearman']:+.2f}")
    print("\nTree rules:\n" + report["tree_rules"])
    print("Cluster curve:", cluster_curve)
    print("\nHardest 20:")
    for r in report["hardest"]:
        print(f"  {r['word']:6s} {r['difficulty']:.2f}  cluster={r['cluster_size']:2d} {r['cluster_pattern']:6s} opener_bucket={r['opener_bucket']:.1f}")


if __name__ == "__main__":
    main()
