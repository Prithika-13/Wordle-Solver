import numpy as np

from .base import Solver


class RandomSolver(Solver):
    """Guess a uniformly random word among those still consistent with the feedback."""

    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def choose(self, cand, history, turn):
        return int(self.ans_as_guess[self.rng.choice(cand)])
