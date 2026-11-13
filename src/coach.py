"""Post-game coach: grade every guess of a finished (or in-progress) Wordle game.

For each guess we compare what the player did with what the information-theoretic solver would do:

* expected bits  - how much the guess is expected to reveal, averaged over the answers still possible
* best bits      - the expected bits of the best available guess in that position
* actual bits    - how much the guess really revealed, log2(answers before / answers after)
* efficiency     - expected / best      -> SKILL  (did you choose a good question?)
* luck           - actual - expected    -> LUCK   (did the feedback happen to be kind?)

Skill is about the decision, luck is about the outcome, so the two are reported separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.feedback import ALL_GREEN, decode, encode
from src.solvers.entropy import EntropySolver, entropy_scores

EXCELLENT, GOOD, OKAY = 0.97, 0.85, 0.65


@dataclass
class TurnReport:
    turn: int
    word: str
    pattern: str
    n_before: int
    n_after: int
    expected_bits: float
    best_bits: float
    best_word: str
    actual_bits: float
    efficiency: float  # 0..1, skill of the choice
    luck: float  # bits, + means luckier than expected
    percentile: float  # % of all allowed guesses that are expected to reveal less than this one
    could_win: bool  # the guess was still a possible answer
    rating: str
    note: str


def rating_for(eff: float) -> str:
    return "Excellent" if eff >= EXCELLENT else "Good" if eff >= GOOD else "Okay" if eff >= OKAY else "Weak"


def parse_pattern(text: str) -> str:
    """Accept G/Y/- , 2/1/0 or the emoji squares; return a canonical 5-char G/Y/- pattern."""
    m = {"G": "G", "Y": "Y", "-": "-", "B": "-", "X": "-", "_": "-", "2": "G", "1": "Y", "0": "-",
         "🟩": "G", "🟨": "Y", "⬛": "-", "⬜": "-", "🟫": "-"}
    out = [m[c.upper() if c.isascii() else c] for c in text.strip() if not c.isspace()]
    if len(out) != 5:
        raise ValueError("a pattern needs exactly 5 squares")
    return "".join(out)


def parse_game(text: str) -> list[tuple[str, str]]:
    """Parse lines like 'crane -Y--G' (word then colours) into [(word, pattern)]."""
    game = []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"can't read line: {ln!r} (expected: word colours)")
        game.append((parts[0].lower(), parse_pattern(parts[1])))
    return game


def analyze(solver: EntropySolver, history: list[tuple[str, str]]) -> dict:
    """Grade a game. history = [(word, 'G--Y-'), ...]. Raises ValueError for inconsistent input."""
    M, answers, guesses = solver.M, solver.answers, solver.guesses
    gi = solver.guess_index
    cand = np.arange(len(answers))
    turns: list[TurnReport] = []
    for t, (word, pat) in enumerate(history, start=1):
        if word not in gi:
            raise ValueError(f"'{word}' is not a valid Wordle word")
        g, code = gi[word], encode(pat)
        n_before = len(cand)
        is_cand = np.zeros(len(guesses), bool)
        is_cand[solver.ans_as_guess[cand]] = True
        after = cand[M[g, cand] == code]
        if len(after) == 0:
            why = ("no answer would give those colours" if t == 1 else "it contradicts the earlier feedback")
            raise ValueError(f"guess {t} ({word}, {pat}) is impossible: {why}")
        n_after = len(after)
        actual = float(np.log2(n_before / n_after))

        if n_before == 1:  # nothing left to learn: the only good move is the answer itself
            best_g = int(solver.ans_as_guess[cand[0]])
            exp_bits = best_bits = 0.0
            eff = 1.0 if g == best_g else 0.0
            pct = 100.0 if g == best_g else 0.0
            note = "Only one word was left, so this was the winning move." if eff else "Only one word was left; guessing it wins."
        else:
            h = entropy_scores(M, cand)
            score = h + is_cand / n_before
            best_g = int(np.argmax(score))
            exp_bits, best_bits = float(h[g]), float(h[best_g])
            eff = exp_bits / best_bits if best_bits > 0 else 1.0
            pct = float(100 * (h < exp_bits - 1e-9).mean())
            note = _note(n_before, is_cand[g], is_cand[best_g], eff, guesses[best_g])
            if n_before <= 2 and is_cand[g]:  # any consistent guess is optimal with <=2 words left
                eff = 1.0
        eff = float(min(1.0, eff))
        turns.append(TurnReport(
            turn=t, word=word, pattern=pat, n_before=n_before, n_after=n_after, expected_bits=exp_bits,
            best_bits=best_bits, best_word=guesses[best_g], actual_bits=actual, efficiency=eff,
            luck=actual - exp_bits, percentile=pct, could_win=bool(is_cand[g]), rating=rating_for(eff), note=note))
        cand = after
        if code == ALL_GREEN:
            cand = np.array([solver.answers.index(word)])

    solved = bool(history) and history[-1][1] == "GGGGG"
    graded = [x for x in turns if x.n_before >= 2]
    tot_best = sum(x.best_bits for x in graded)
    skill = sum(x.expected_bits for x in graded) / tot_best if tot_best > 0 else 1.0
    out = dict(
        turns=[asdict(x) for x in turns],
        solved=solved,
        n_guesses=len(history),
        skill=float(min(1.0, skill)),
        luck_bits=float(sum(x.luck for x in turns if x.n_before >= 2)),
        remaining=len(cand),
        biggest_miss=None,
        luckiest=None,
        solver_game=None,
    )
    if graded:
        miss = max(graded, key=lambda x: x.best_bits - x.expected_bits)
        if miss.best_bits - miss.expected_bits > 0.05:
            out["biggest_miss"] = asdict(miss)
        out["luckiest"] = asdict(max(graded, key=lambda x: x.luck))
        out["unluckiest"] = asdict(min(graded, key=lambda x: x.luck))
    if solved:  # replay: how would the entropy solver have played the same answer?
        ai = answers.index(history[-1][0])
        gs = solver.play(ai, max_turns=10)
        replay = []
        for g in gs:
            c = int(M[g, ai])
            replay.append((guesses[g], decode(c)))
        out["solver_game"] = replay
    out["verdict"] = verdict(out)
    return out


def _note(n_before: int, played_cand: bool, best_cand: bool, eff: float, best_word: str) -> str:
    if eff >= EXCELLENT:
        return "A near-perfect question for this position."
    if not played_cand and best_cand and n_before <= 8:
        return f"With only {n_before} words left, guessing a possible answer such as {best_word.upper()} lets you win and still learn."
    if played_cand and not best_cand and n_before >= 20:
        return f"A possible answer, but {best_word.upper()} splits the {n_before:,} remaining words more evenly."
    if eff < OKAY:
        return f"Many of the {n_before:,} remaining words would give you the same colours; {best_word.upper()} separates them better."
    return f"Fine, though {best_word.upper()} would reveal a bit more on average."


def verdict(rep: dict) -> str:
    sk, n = rep["skill"], rep["n_guesses"]
    if not rep["turns"]:
        return "Add at least one guess to get a report."
    skill_txt = ("excellent" if sk >= 0.95 else "good" if sk >= 0.85 else "decent" if sk >= 0.7 else "rough")
    luck = rep["luck_bits"]
    luck_txt = ("very good luck" if luck > 1.5 else "a little good luck" if luck > 0.5 else
                "very bad luck" if luck < -1.5 else "a little bad luck" if luck < -0.5 else "about average luck")
    s = f"Your guessing skill was {skill_txt} ({sk:.0%} of the best possible information) and you had {luck_txt} ({luck:+.1f} bits)."
    if rep["solver_game"]:
        ns = len(rep["solver_game"])
        s += (f" The entropy solver needs {ns} guesses on this answer; you used {n}."
              if ns != n else f" The entropy solver also needs {ns} guesses on this answer.")
    return s
