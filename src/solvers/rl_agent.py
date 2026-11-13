"""Policy network + RLSolver (greedy evaluation of a trained policy)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.features import FeatureBuilder

from .base import Solver


class WordPolicy(nn.Module):
    """Scores every word with a shared MLP on its feature vector (word-agnostic => generalises).

    obs: [B, V, F] -> logits [B, V], value [B]
    """

    def __init__(self, n_input: int, hidden: int = 128):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(n_input, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.v = nn.Sequential(nn.Linear(2 * n_input, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor):
        logits = self.pi(obs).squeeze(-1)
        pooled = torch.cat([obs.mean(1), obs.max(1).values], dim=-1)
        return logits, self.v(pooled).squeeze(-1)


def save_policy(net: WordPolicy, path, mode: str, n_input: int, hidden: int, k: int, extra=None):
    torch.save(dict(state=net.state_dict(), mode=mode, n_input=n_input, hidden=hidden, k=k, extra=extra or {}), path)


def load_policy(path) -> tuple[WordPolicy, str, int]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = WordPolicy(ck["n_input"], ck["hidden"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["mode"], ck["k"]


class RLSolver(Solver):
    """Greedy (argmax) play of a trained policy."""

    def __init__(self, path: str | Path, name: str | None = None):
        super().__init__()
        self.net, self.mode, k = load_policy(path)
        self.fb = FeatureBuilder(mode=self.mode, k=k)
        self.name = name or f"rl-{self.mode}"

    @torch.no_grad()
    def choose(self, cand, history, turn):
        obs, sel = self.fb.state(cand, turn, key=history)
        logits, _ = self.net(torch.from_numpy(obs)[None])
        return int(self.fb.pool[sel[int(logits.argmax(-1))]])
