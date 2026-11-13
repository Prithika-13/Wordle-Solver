"""Gymnasium environment for Wordle.

Action  : index 0..K-1 into a shortlist of the K most promising guess words (see features.state).
Obs     : float32 [K, n_input] per-word feature matrix of that shortlist.
Reward  : -1 per guess; failing after 6 guesses costs an extra `fail_penalty`.
          Optional potential-based shaping rewards shrinking the candidate set.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.feedback import ALL_GREEN
from src.features import FeatureBuilder


class WordleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        fb: FeatureBuilder | None = None,
        answer_pool: np.ndarray | None = None,
        mode: str = "assisted",
        max_turns: int = 6,
        fail_penalty: float = 4.0,
        shaping: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.fb = fb or FeatureBuilder(mode=mode)
        self.na = len(self.fb.answers)
        self.answer_pool = np.arange(self.na) if answer_pool is None else np.asarray(answer_pool)
        self.max_turns = max_turns
        self.fail_penalty = fail_penalty
        self.shaping = shaping
        self.action_space = spaces.Discrete(self.fb.k)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.fb.k, self.fb.n_input), np.float32)
        self._sel = np.zeros(self.fb.k, dtype=int)
        self.rng = np.random.default_rng(seed)
        self.answer = 0
        self.cand = np.arange(self.na)
        self.turn = 0
        self.history: tuple = ()

    # ---- gym API -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "answer" in options:
            self.answer = int(options["answer"])
        else:
            self.answer = int(self.rng.choice(self.answer_pool))
        self.cand = np.arange(self.na)
        self.turn = 0
        self.history = ()
        return self._obs(), {"answer": self.answer}

    def step(self, action: int):
        g = int(self.fb.pool[self._sel[int(action)]])
        code = int(self.fb.M[g, self.answer])
        solved = code == ALL_GREEN
        n_before = len(self.cand)
        self.turn += 1
        reward = -1.0
        if solved:
            if self.shaping:  # potential-based: candidate set collapses to 1 word
                reward += self.shaping * np.log2(n_before)
        else:
            self.history += ((g, code),)
            self.cand = self.cand[self.fb.M[g, self.cand] == code]
            if self.shaping:
                reward += self.shaping * (np.log2(n_before) - np.log2(len(self.cand)))
        terminated = solved
        truncated = (not solved) and self.turn >= self.max_turns
        if truncated:
            reward -= self.fail_penalty
        obs = self._obs() if not (terminated or truncated) else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, truncated, {"solved": solved, "n_cand": len(self.cand)}

    def _obs(self):
        obs, self._sel = self.fb.state(self.cand, self.turn, key=self.history)
        return obs
