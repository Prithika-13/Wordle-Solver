import numpy as np

from src.feedback import _to_arr

from .base import Solver


class FrequencyHeuristic(Solver):
    """Pick the remaining candidate with the highest letter-coverage score.

    Score = sum over distinct letters of how many remaining candidates contain that letter.
    """

    name = "frequency"

    def __init__(self):
        super().__init__()
        A = _to_arr(self.answers) - 97
        self.present = np.zeros((len(self.answers), 26), dtype=np.int32)
        for p in range(5):
            self.present[np.arange(len(A)), A[:, p]] = 1
        self._memo: dict = {}

    def choose(self, cand, history, turn):
        if history in self._memo:
            return self._memo[history]
        counts = self.present[cand].sum(axis=0)
        scores = (self.present[cand] * counts).sum(axis=1)
        g = int(self.ans_as_guess[cand[np.argmax(scores)]])
        self._memo[history] = g
        return g
