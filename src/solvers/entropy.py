from __future__ import annotations

import numpy as np

from .base import Solver

LOG2 = np.log(2.0)


def entropy_scores(M: np.ndarray, cand: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Expected information (bits) of every guess, given uniform prior over `cand` answers.

    Returns array of shape [n_guesses].
    """
    ng = M.shape[0]
    n = len(cand)
    out = np.empty(ng, dtype=np.float64)
    for s in range(0, ng, chunk):
        sub = M[s : s + chunk][:, cand].astype(np.int32)
        k = sub.shape[0]
        sub += (np.arange(k, dtype=np.int32) * 243)[:, None]
        counts = np.bincount(sub.ravel(), minlength=k * 243).reshape(k, 243)
        p = counts / n
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1) / LOG2
        out[s : s + k] = h
    return out


class EntropySolver(Solver):
    """Greedy one-step information maximiser.

    candidates_only: restrict guesses to words that could still be the answer.
    candidate_bonus: extra score (in bits, scaled by 1/n_candidates) for guessing a possible answer;
        breaks entropy ties in favour of guesses that can win immediately.
    """

    def __init__(self, candidates_only: bool = False, candidate_bonus: float = 1.0):
        super().__init__()
        self.candidates_only = candidates_only
        self.candidate_bonus = candidate_bonus
        self._memo: dict = {}
        self.name = "entropy" + ("-cand" if candidates_only else "")

    def score_guesses(self, cand: np.ndarray) -> np.ndarray:
        """Score for every guess (higher is better)."""
        n = len(cand)
        h = entropy_scores(self.M, cand)
        is_cand = np.zeros(len(self.guesses), dtype=bool)
        is_cand[self.ans_as_guess[cand]] = True
        score = h + self.candidate_bonus * is_cand / n
        if self.candidates_only:
            score = np.where(is_cand, score, -np.inf)
        return score

    def choose(self, cand, history, turn):
        if history in self._memo:
            return self._memo[history]
        if len(cand) <= 2:
            g = int(self.ans_as_guess[cand[0]])  # guess a candidate: can't do better
        else:
            g = int(np.argmax(self.score_guesses(cand)))
        self._memo[history] = g
        return g

    def top_k(self, cand: np.ndarray, k: int = 5) -> list[tuple[str, float, bool]]:
        """Top-k guesses as (word, expected_bits, is_candidate)."""
        h = entropy_scores(self.M, cand)
        score = self.score_guesses(cand)
        is_cand = np.zeros(len(self.guesses), dtype=bool)
        is_cand[self.ans_as_guess[cand]] = True
        idx = np.argsort(-score)[:k]
        return [(self.guesses[i], float(h[i]), bool(is_cand[i])) for i in idx]
