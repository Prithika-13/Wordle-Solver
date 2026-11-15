"""Tools the coach LLM can call. Every number the LLM states must come from one of these.

Each tool wraps code that already exists (entropy scoring, the coach grader, the candidate filter, the
difficulty table), so the language model never computes or recalls a figure: it only chooses which tool
to call and explains the result.

Positions: ``after_turn`` is how many of the player's guesses have been played. 0 = the opening position,
``len(history)`` (the default) = the position where the player's *next* guess would be made.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.coach import analyze
from src.feedback import decode, encode
from src.memory import PlayerMemory
from src.solvers.entropy import EntropySolver, entropy_scores

_AFTER_TURN = {
    "type": "integer",
    "description": "How many of the player's guesses have been played: 0 = the opening position. "
                   "Omit for the position after all guesses so far.",
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_game_report",
        "description": "Grade the player's current game. Returns, for every guess: answers left before/after, expected bits, "
                       "the best available guess and its expected bits, the bits actually revealed, skill efficiency (0-1), luck "
                       "(bits), rating and percentile. Also the overall skill, total luck, the verdict and, for a solved game, "
                       "how the entropy solver would have played the same answer. Call this first for any question about the "
                       "player's own guesses.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "score_guess",
        "description": "How good is a specific word as a guess at a given position of the player's game? Returns the expected "
                       "bits of information, whether it could still be the answer, how it compares with the best available "
                       "guess, and its percentile among all allowed words. Use it to compare alternatives such as 'would CRANE "
                       "have been better?'.",
        "input_schema": {
            "type": "object",
            "properties": {"word": {"type": "string", "description": "A 5-letter word."}, "after_turn": _AFTER_TURN},
            "required": ["word"],
            "additionalProperties": False,
        },
    },
    {
        "name": "best_guesses",
        "description": "The top guesses at a position of the player's game, ranked by expected information, with the number "
                       "of answers still possible. Use it to answer 'what should I play here?' or 'what is the best opener?'.",
        "input_schema": {
            "type": "object",
            "properties": {"after_turn": _AFTER_TURN, "k": {"type": "integer", "description": "How many to return (1-10, default 5)."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "remaining_answers",
        "description": "How many answers are still possible at a position of the player's game, and (up to a limit) which ones.",
        "input_schema": {
            "type": "object",
            "properties": {"after_turn": _AFTER_TURN, "limit": {"type": "integer", "description": "Max words to list (0-30, default 15)."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "solve_with_entropy_solver",
        "description": "Play a full game of the entropy solver against a given answer word and return its guesses with the "
                       "colour pattern of each (G green, Y yellow, - gray). Use it to show how the solver would handle a word.",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string", "description": "A possible Wordle answer (5 letters)."}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_player_history",
        "description": "The player's saved history across past games (memory): number of games, average skill and luck, how often "
                       "they solve, their usual openers and how good each is, which guess number they play worst, weak words they "
                       "repeat, a skill trend (only with 6+ games) and their most recent games. Use it for questions about patterns, "
                       "progress or habits. The statistics are computed from games the player chose to save; say how many games "
                       "they are based on and avoid conclusions from fewer than 3 games.",
        "input_schema": {
            "type": "object",
            "properties": {"last_n": {"type": "integer", "description": "How many recent games to list (1-10, default 5)."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "word_difficulty",
        "description": "How hard is a word for the solvers, and why? Returns its average guesses, difficulty rank out of all "
                       "answers, look-alike cluster (e.g. _ATCH with the other words in it) and letter-rarity features.",
        "input_schema": {
            "type": "object",
            "properties": {"word": {"type": "string", "description": "A possible Wordle answer (5 letters)."}},
            "required": ["word"],
            "additionalProperties": False,
        },
    },
]


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


class CoachTools:
    """Executes the coach's tool calls against one game."""

    def __init__(self, solver: EntropySolver, history: list[tuple[str, str]], difficulty: pd.DataFrame | None = None,
                 memory: PlayerMemory | None = None):
        self.s = solver
        self.memory = memory
        self.history = [tuple(x) for x in history]
        self.difficulty = difficulty
        self._answers = set(solver.answers)

    # -- position helpers ---------------------------------------------------------------------
    def _position(self, after_turn: Any) -> tuple[int, np.ndarray]:
        n = len(self.history)
        t = n if after_turn is None else after_turn
        if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t <= n:
            raise ValueError(f"after_turn must be an integer from 0 to {n}")
        cand = np.arange(len(self.s.answers))
        for word, pat in self.history[:t]:
            cand = cand[self.s.M[self.s.guess_index[word], cand] == encode(pat)]
        return t, cand

    def _scores(self, cand: np.ndarray):
        h = entropy_scores(self.s.M, cand)
        is_cand = np.zeros(len(self.s.guesses), bool)
        is_cand[self.s.ans_as_guess[cand]] = True
        return h, is_cand

    @staticmethod
    def _word(value: Any) -> str:
        if not isinstance(value, str) or len(value.strip()) != 5 or not value.strip().isalpha():
            raise ValueError("expected a 5-letter word")
        return value.strip().lower()

    # -- tools --------------------------------------------------------------------------------
    def get_game_report(self) -> dict:
        if not self.history:
            return {"error": "The player has not entered any guesses yet.",
                    "hint": "best_guesses(after_turn=0) gives opener advice."}
        rep = analyze(self.s, list(self.history))
        keys = ["turn", "word", "pattern", "n_before", "n_after", "expected_bits", "best_word", "best_bits", "actual_bits",
                "efficiency", "luck", "percentile", "could_win", "rating"]
        turns = [{k: (_r(t[k]) if isinstance(t[k], float) else t[k]) for k in keys} for t in rep["turns"]]
        out = {"solved": rep["solved"], "guesses_used": rep["n_guesses"], "skill": _r(rep["skill"]),
               "luck_bits": _r(rep["luck_bits"]), "answers_remaining": rep["remaining"], "verdict": rep["verdict"], "turns": turns}
        if rep["solver_game"]:
            out["entropy_solver_game"] = [{"word": w, "pattern": p} for w, p in rep["solver_game"]]
        return out

    def score_guess(self, word: Any, after_turn: Any = None) -> dict:
        w = self._word(word)
        if w not in self.s.guess_index:
            raise ValueError(f"'{w}' is not an allowed Wordle guess")
        t, cand = self._position(after_turn)
        n = len(cand)
        g = self.s.guess_index[w]
        h, is_cand = self._scores(cand)
        best = int(np.argmax(h + is_cand / n))
        eff = float(h[g] / h[best]) if h[best] > 0 else 1.0
        return {"word": w, "after_turn": t, "answers_remaining": n, "expected_bits": _r(h[g]),
                "could_be_the_answer": bool(is_cand[g]), "best_word": self.s.guesses[best], "best_bits": _r(h[best]),
                "efficiency_vs_best": _r(min(1.0, eff)), "percentile_among_all_words": _r(100 * (h < h[g] - 1e-9).mean(), 1)}

    def best_guesses(self, after_turn: Any = None, k: Any = 5) -> dict:
        t, cand = self._position(after_turn)
        k = 5 if k is None else k
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            raise ValueError("k must be an integer from 1 to 10")
        n = len(cand)
        h, is_cand = self._scores(cand)
        top = np.argsort(-(h + is_cand / n))[:k]
        return {"after_turn": t, "answers_remaining": n,
                "best": [{"word": self.s.guesses[i], "expected_bits": _r(h[i]), "could_be_the_answer": bool(is_cand[i])} for i in top]}

    def remaining_answers(self, after_turn: Any = None, limit: Any = 15) -> dict:
        t, cand = self._position(after_turn)
        limit = 15 if limit is None else limit
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 30:
            raise ValueError("limit must be an integer from 0 to 30")
        words = [self.s.answers[i] for i in cand[:limit]]
        return {"after_turn": t, "answers_remaining": len(cand), "words": words, "list_truncated": len(cand) > limit}

    def solve_with_entropy_solver(self, answer: Any) -> dict:
        a = self._word(answer)
        if a not in self._answers:
            raise ValueError(f"'{a}' is not in the answer list")
        ai = self.s.answers.index(a)
        gs = self.s.play(ai, max_turns=10)
        return {"answer": a, "guesses_needed": len(gs),
                "game": [{"word": self.s.guesses[g], "pattern": decode(int(self.s.M[g, ai]))} for g in gs]}

    def get_player_history(self, last_n: Any = 5) -> dict:
        last_n = 5 if last_n is None else last_n
        if isinstance(last_n, bool) or not isinstance(last_n, int) or not 1 <= last_n <= 10:
            raise ValueError("last_n must be an integer from 1 to 10")
        prof = self.memory.profile(last_n) if self.memory else None
        if prof is None:
            return {"games_saved": 0, "note": "No games have been saved yet, so there is no history to draw on. "
                                              "The player can save a finished game from the Coach report tab."}
        return prof

    def word_difficulty(self, word: Any) -> dict:
        w = self._word(word)
        if self.difficulty is None:
            raise ValueError("the difficulty table is not available (run scripts/difficulty.py)")
        row = self.difficulty[self.difficulty.word == w]
        if row.empty:
            raise ValueError(f"'{w}' is not in the answer list")
        r = row.iloc[0]
        rank = int((self.difficulty.difficulty > r.difficulty).sum()) + 1
        return {"word": w, "avg_guesses": _r(r.difficulty), "difficulty_rank": rank, "of": int(len(self.difficulty)),
                "look_alike_cluster": (str(r.cluster_pattern) or None), "look_alike_cluster_size": int(r.cluster_size),
                "look_alikes": [m for m in str(r.cluster_members).split(", ") if m and m != "nan"][:10],
                "repeated_letters": int(r.dup_letters), "rarest_letter_appears_in_pct_of_answers": _r(100 * r.min_letter_freq, 1)}

    # -- dispatcher ---------------------------------------------------------------------------
    def run(self, name: str, tool_input: dict | None) -> tuple[str, bool]:
        """Execute a tool. Returns (json_string, is_error). Never raises: bad input becomes an error result."""
        fn = getattr(self, name, None)
        if name not in {t["name"] for t in TOOL_SCHEMAS} or fn is None:
            return json.dumps({"error": f"unknown tool '{name}'"}), True
        try:
            return json.dumps(fn(**(tool_input or {}))), False
        except (ValueError, TypeError, KeyError) as e:
            return json.dumps({"error": str(e)}), True
