"""Evaluate solvers over all answers.  Usage: python -m src.evaluate [random|frequency|entropy|...]"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"


def evaluate(solver, answer_idx=None, max_turns: int = 6) -> pd.DataFrame:
    """Play every answer. Failed games count as max_turns+1 guesses in `score`."""
    if answer_idx is None:
        answer_idx = range(len(solver.answers))
    rows = []
    for j in answer_idx:
        g = solver.play(j, max_turns)
        ok = bool(solver.solved(g, j))
        rows.append(
            dict(
                solver=solver.name,
                answer=solver.answers[j],
                solved=ok,
                n_guesses=len(g),
                score=len(g) if ok else max_turns + 1,
                guesses=" ".join(solver.guesses[i] for i in g),
            )
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    solved = df[df.solved]
    dist = Counter(df.n_guesses.where(df.solved, 0))
    return dict(
        solver=df.solver.iloc[0],
        games=len(df),
        solve_rate=df.solved.mean(),
        mean_guesses_solved=solved.n_guesses.mean(),
        mean_score=df.score.mean(),
        median=df.score.median(),
        max=int(df.score.max()),
        opener=df.guesses.iloc[0].split()[0],
        **{f"n{k}": dist.get(k, 0) for k in range(1, 7)},
        fails=dist.get(0, 0),
    )


def make_solver(name: str):
    from src.solvers.entropy import EntropySolver
    from src.solvers.heuristic import FrequencyHeuristic
    from src.solvers.random_solver import RandomSolver

    return {
        "random": RandomSolver,
        "frequency": FrequencyHeuristic,
        "entropy": EntropySolver,
        "entropy-cand": lambda: EntropySolver(candidates_only=True),
    }[name]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solvers", nargs="*", default=["random", "frequency", "entropy"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    summaries = []
    for name in args.solvers:
        s = make_solver(name)
        t0 = time.time()
        df = evaluate(s)
        dt = time.time() - t0
        df.to_csv(RESULTS / f"eval_{s.name}.csv", index=False)
        sm = summarize(df)
        sm["seconds"] = round(dt, 1)
        summaries.append(sm)
        print(f"{s.name:14s} mean={sm['mean_score']:.4f} solve={sm['solve_rate']:.4f} opener={sm['opener']} ({dt:.0f}s)")
    out = pd.DataFrame(summaries)
    print(out.to_string(index=False))
    out.to_csv(RESULTS / "summary_baselines.csv", index=False)


if __name__ == "__main__":
    main()
