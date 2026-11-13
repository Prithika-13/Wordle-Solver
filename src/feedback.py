"""Wordle feedback, word lists, and the precomputed feedback matrix.

Feedback is encoded as a base-3 integer in [0, 242]: per position
0 = gray, 1 = yellow, 2 = green; position 0 is the least significant digit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "feedback_matrix.npy"
ALL_GREEN = 3**5 - 1  # 242


def load_words() -> tuple[list[str], list[str]]:
    """Return (answers, guesses). `guesses` is a superset of `answers`."""
    answers = (DATA_DIR / "answers.txt").read_text().split()
    guesses = (DATA_DIR / "allowed.txt").read_text().split()
    return answers, guesses


def feedback(guess: str, answer: str) -> int:
    """Reference implementation with correct duplicate-letter handling."""
    code = [0] * 5
    remaining: dict[str, int] = {}
    for g, a in zip(guess, answer):  # greens first
        if g == a:
            continue
        remaining[a] = remaining.get(a, 0) + 1
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            code[i] = 2
        elif remaining.get(g, 0) > 0:  # yellows only up to remaining count
            code[i] = 1
            remaining[g] -= 1
    return sum(c * 3**i for i, c in enumerate(code))


def decode(code: int) -> str:
    """243-code -> string like 'GY--G' (G green, Y yellow, - gray)."""
    out = []
    for _ in range(5):
        out.append("-YG"[code % 3])
        code //= 3
    return "".join(out)


def encode(pattern: str) -> int:
    """'GY--G' (also accepts 2/1/0) -> code."""
    m = {"G": 2, "Y": 1, "-": 0, "B": 0, "X": 0, "2": 2, "1": 1, "0": 0}
    return sum(m[c.upper()] * 3**i for i, c in enumerate(pattern))


def _to_arr(words: list[str]) -> np.ndarray:
    return np.frombuffer("".join(words).encode(), dtype=np.uint8).reshape(-1, 5)


def compute_matrix(guesses: list[str], answers: list[str]) -> np.ndarray:
    """Vectorised [n_guesses, n_answers] uint8 feedback matrix."""
    G, A = _to_arr(guesses), _to_arr(answers)
    ng, na = len(G), len(A)
    greens = G[:, None, :] == A[None, :, :]  # [ng, na, 5]
    # letters of the answer not consumed by a green: counts per letter
    out = np.zeros((ng, na), dtype=np.uint8)
    # remaining[g, a, letter] = count of letter in answer at non-green positions
    remaining = np.zeros((ng, na, 26), dtype=np.int8)
    for p in range(5):
        letter = (A[:, p] - 97)[None, :]
        mask = ~greens[:, :, p]
        gi, ai = np.nonzero(mask)
        np.add.at(remaining, (gi, ai, np.broadcast_to(letter, (ng, na))[gi, ai]), 1)
    gidx, aidx = np.meshgrid(np.arange(ng), np.arange(na), indexing="ij")
    for p in range(5):
        gl = (G[:, p] - 97)[:, None]
        gl = np.broadcast_to(gl, (ng, na))
        is_green = greens[:, :, p]
        has = remaining[gidx, aidx, gl] > 0
        yellow = ~is_green & has
        # consume the yellow
        yi, yj = np.nonzero(yellow)
        remaining[yi, yj, gl[yi, yj]] -= 1
        digit = np.where(is_green, 2, np.where(yellow, 1, 0)).astype(np.uint8)
        out += digit * np.uint8(3**p)
    return out


_cache: dict[str, object] = {}


def get_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    """Return (matrix[guesses, answers], answers, guesses); cached on disk."""
    if "m" in _cache:
        return _cache["m"]  # type: ignore[return-value]
    answers, guesses = load_words()
    if CACHE.exists():
        m = np.load(CACHE)
        if m.shape != (len(guesses), len(answers)):
            m = None
    else:
        m = None
    if m is None:
        m = compute_matrix(guesses, answers)
        np.save(CACHE, m)
    _cache["m"] = (m, answers, guesses)
    return _cache["m"]  # type: ignore[return-value]
