"""Player memory: a small local store of graded games, plus statistics computed from it.

Design choices
* Memory holds only *computed* facts (the grader's output for each saved game), never text written by an LLM,
  so the coach can't "remember" something that didn't happen and stored text can't carry injected instructions.
* It is a plain JSON file you can read, download or delete. Default: <repo>/data/player_history.json (git-ignored),
  overridable with WORDLE_MEMORY_PATH.
* Saving is explicit (a button in the app) and idempotent: the same game is never stored twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = 1
TURN_KEYS = ["turn", "word", "pattern", "n_before", "expected_bits", "best_bits", "best_word", "efficiency", "luck", "rating", "could_win"]


def default_path() -> Path:
    return Path(os.environ.get("WORDLE_MEMORY_PATH") or ROOT / "data" / "player_history.json")


def game_id(history: list[tuple[str, str]]) -> str:
    return hashlib.sha1("|".join(f"{w}:{p}" for w, p in history).encode()).hexdigest()[:12]


def _avg(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def is_finished(rep: dict) -> bool:
    """A game worth remembering: solved, or all six guesses used."""
    return bool(rep.get("solved")) or rep.get("n_guesses", 0) >= 6


class PlayerMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_path()

    # -- storage ------------------------------------------------------------------------------
    def games(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text())
            return list(data.get("games", [])) if isinstance(data, dict) else []
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError):
            # unreadable file: keep a copy instead of silently overwriting it on the next save
            try:
                self.path.replace(self.path.with_suffix(".corrupt.json"))
            except OSError:
                pass
            return []

    def _write(self, games: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"version": VERSION, "games": games}, f, indent=1)
            os.replace(tmp, self.path)  # atomic: a crash can't leave a half-written file
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def has(self, history: list[tuple[str, str]]) -> bool:
        gid = game_id(history)
        return any(g["id"] == gid for g in self.games())

    def save_game(self, rep: dict, source: str = "played") -> dict:
        """Store a graded game (the dict returned by src.coach.analyze). Returns {"saved", "duplicate", "id"}."""
        turns = rep["turns"]
        if not turns:
            raise ValueError("nothing to save: the game has no guesses")
        hist = [(t["word"], t["pattern"]) for t in turns]
        gid = game_id(hist)
        games = self.games()
        if any(g["id"] == gid for g in games):
            return {"saved": False, "duplicate": True, "id": gid}
        rec = {
            "id": gid,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "solved": bool(rep["solved"]),
            "answer": turns[-1]["word"] if rep["solved"] else None,
            "n_guesses": rep["n_guesses"],
            "skill": round(rep["skill"], 4),
            "luck_bits": round(rep["luck_bits"], 3),
            "opener": turns[0]["word"],
            "solver_guesses": len(rep["solver_game"]) if rep.get("solver_game") else None,
            "turns": [{k: (round(t[k], 3) if isinstance(t[k], float) else t[k]) for k in TURN_KEYS} for t in turns],
        }
        games.append(rec)
        self._write(games)
        return {"saved": True, "duplicate": False, "id": gid}

    def clear(self) -> int:
        n = len(self.games())
        if self.path.exists():
            self.path.unlink()
        return n

    # -- statistics ---------------------------------------------------------------------------
    def profile(self, last_n: int = 5) -> dict | None:
        """Everything the coach may say about the player's history. None when nothing is saved."""
        games = self.games()
        if not games:
            return None
        n = len(games)
        graded = [t for g in games for t in g["turns"] if t["n_before"] >= 2]

        by_open: dict[str, list[dict]] = defaultdict(list)
        for g in games:
            by_open[g["opener"]].append(g["turns"][0])
        openers = sorted(
            ({"word": w, "games": len(ts), "avg_efficiency": _avg(t["efficiency"] for t in ts)} for w, ts in by_open.items()),
            key=lambda d: (-d["games"], d["word"]),
        )[:5]

        by_turn: dict[int, list[dict]] = defaultdict(list)
        for t in graded:
            by_turn[t["turn"]].append(t)
        guess_stats = [
            {"guess_number": k, "n": len(ts), "avg_efficiency": _avg(t["efficiency"] for t in ts),
             "avg_bits_lost": _avg(t["best_bits"] - t["expected_bits"] for t in ts)}
            for k, ts in sorted(by_turn.items())
        ]
        eligible = [s for s in guess_stats if s["n"] >= 3]
        weakest = min(eligible, key=lambda s: s["avg_efficiency"]) if eligible else None

        weak_words = Counter(t["word"] for t in graded if t["rating"] == "Weak")
        trend = None
        if n >= 6:
            skills = [g["skill"] for g in games]
            recent, earlier = _avg(skills[-3:]), _avg(skills[:-3])
            diff = recent - earlier
            trend = {"recent_3_avg_skill": recent, "earlier_avg_skill": earlier,
                     "direction": "improving" if diff > 0.03 else "declining" if diff < -0.03 else "flat"}
        return {
            "games_saved": n,
            "solved_rate": _avg(1.0 if g["solved"] else 0.0 for g in games),
            "avg_guesses": _avg(g["n_guesses"] for g in games),
            "avg_skill": _avg(g["skill"] for g in games),
            "avg_luck_bits": _avg(g["luck_bits"] for g in games),
            "share_of_guesses_rated_weak_or_okay": _avg(1.0 if t["rating"] in ("Weak", "Okay") else 0.0 for t in graded),
            "skill_trend": trend,
            "openers": openers,
            "by_guess_number": guess_stats,
            "weakest_guess_number": weakest["guess_number"] if weakest else None,
            "weak_words_played_more_than_once": [{"word": w, "times": c} for w, c in weak_words.most_common(5) if c >= 2],
            "guesses_that_could_not_win_with_6_or_fewer_answers_left": sum(
                1 for t in graded if t["n_before"] <= 6 and not t["could_win"]),
            "recent_games": [
                {"date": g["saved_at"][:10], "answer": g["answer"], "guesses": g["n_guesses"], "skill": g["skill"],
                 "luck_bits": g["luck_bits"], "opener": g["opener"], "solver_guesses": g["solver_guesses"]}
                for g in games[-last_n:]
            ],
        }
