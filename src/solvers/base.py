from __future__ import annotations

import numpy as np

from src.feedback import ALL_GREEN, get_matrix


class Solver:
    """A solver picks a guess index (into `guesses`) given the candidate answer indices."""

    name = "base"

    def __init__(self):
        self.M, self.answers, self.guesses = get_matrix()
        self.guess_index = {w: i for i, w in enumerate(self.guesses)}
        # guess index of each answer word (answers are a subset of guesses)
        self.ans_as_guess = np.array([self.guess_index[w] for w in self.answers])

    def choose(self, cand: np.ndarray, history: tuple, turn: int) -> int:
        raise NotImplementedError

    def play(self, answer_idx: int, max_turns: int = 6) -> list[int]:
        """Play one game; returns the guess indices used (last one is the answer if solved)."""
        cand = np.arange(len(self.answers))
        history: tuple = ()
        guesses: list[int] = []
        for turn in range(max_turns):
            g = self.choose(cand, history, turn)
            guesses.append(g)
            code = int(self.M[g, answer_idx])
            if code == ALL_GREEN:
                break
            history += ((g, code),)
            cand = cand[self.M[g, cand] == code]
        return guesses

    def solved(self, guesses: list[int], answer_idx: int) -> bool:
        return self.ans_as_guess[answer_idx] == guesses[-1]
