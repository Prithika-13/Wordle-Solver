"""Evaluate every solver on the held-out test answers (and on all answers) and write a summary table.

    python scripts/compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.train_rl import split_answers
from src.evaluate import evaluate, make_solver, summarize
from src.feedback import load_words

RES = ROOT / "results"


def main():
    answers, _ = load_words()
    train_idx, test_idx = split_answers(len(answers))
    solvers = [make_solver(n) for n in ["random", "frequency", "entropy-cand", "entropy"]]
    from src.solvers.rl_agent import RLSolver

    for ckpt in sorted(RES.glob("rl_*.pt")):
        solvers.append(RLSolver(ckpt, name=ckpt.stem))
    rows, frames = [], []
    for s in solvers:
        for split, idx in [("test", test_idx), ("train", train_idx)]:
            df = evaluate(s, idx)
            df["split"] = split
            frames.append(df)
            sm = summarize(df)
            sm["split"] = split
            rows.append(sm)
            print(f"{s.name:18s} {split:5s} mean={sm['mean_score']:.3f} solve={sm['solve_rate']:.4f}")
    pd.concat(frames).to_csv(RES / "eval_all.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "summary.csv", index=False)
    print(out[["solver", "split", "games", "solve_rate", "mean_score", "median", "max", "opener"]].to_string(index=False))


if __name__ == "__main__":
    main()
