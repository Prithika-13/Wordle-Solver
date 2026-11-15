"""AI-written coach summary with a grounding check.

The LLM writes the *narrative* (headline, what went well, what to improve, one drill) from facts that code has already
computed. Because a language model can still slip a made-up figure into fluent prose, every number and every
capitalised word in the result is checked against the facts. If something isn't grounded, the model is asked once to
rewrite; whatever is still unverified is reported to the caller so the UI can say so.

Known limits of the check: integers 0-10 are always allowed (they are mostly ordinals and counts such as "guess 3"), and
it verifies that a figure *exists* in the facts, not that it is attached to the right guess.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from src.coach_agent import DEFAULT_MODEL


class CoachSummary(BaseModel):
    headline: str
    went_well: list[str]
    to_improve: list[str]
    next_drill: str


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One sentence verdict on the game, at most 15 words."},
        "went_well": {"type": "array", "items": {"type": "string"}, "description": "1-3 specific things the player did well."},
        "to_improve": {"type": "array", "items": {"type": "string"}, "description": "1-3 specific things to improve."},
        "next_drill": {"type": "string", "description": "One concrete thing to practise next."},
    },
    "required": ["headline", "went_well", "to_improve", "next_drill"],
    "additionalProperties": False,
}

SUMMARY_SYSTEM = """You write a short coaching summary of one Wordle game for the player who just played it. All figures \
have already been computed by code and are given to you as JSON facts. You only explain them.

Hard rules (a program checks them and rejects violations):
- Use ONLY numbers and words that appear in the facts. Copy figures exactly as given (at most 2 decimals; a fraction such as 0.71 \
may be written 71%). Do NOT compute anything new: no sums, differences or ratios. The facts already contain precomputed fields \
such as bits_lost, skill, luck_bits and efficiency: quote those.
- Write a Wordle word in CAPITALS only when it appears in the facts. Do not use capitals for emphasis.
- Skill is the quality of the choices (efficiency vs the best guess); luck is how kind the colours were (luck_bits). Keep them \
separate: never blame a good choice for bad luck or praise luck as skill.
- If `history` is present you may mention at most one pattern from it, only when games_saved is 3 or more, and say how many games \
it is based on. If history_includes_this_game is false, do not treat this game as part of it.

Style: encouraging, specific, concise. headline: one sentence, at most 15 words. went_well and to_improve: 1-3 bullets each, \
at most 25 words per bullet. next_drill: one concrete, actionable suggestion."""


@dataclass
class SummaryResult:
    summary: CoachSummary | None
    verified: bool = False
    problems: list[str] = field(default_factory=list)  # unresolved grounding problems after the last attempt
    attempts: int = 0
    usage: dict = field(default_factory=dict)
    error: str | None = None  # refusal, truncation or schema failure


# ---------------------------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------------------------
def build_facts(game_report: dict, profile: dict | None, this_game_saved: bool = False) -> dict:
    """What the LLM sees. `game_report` is CoachTools.get_game_report(); `profile` is PlayerMemory.profile()."""
    game = copy.deepcopy(game_report)
    for t in game.get("turns", []):  # precompute the one comparison the narrative needs, so the model never subtracts
        t["bits_lost"] = round(max(0.0, t["best_bits"] - t["expected_bits"]), 3) if t["n_before"] >= 2 else 0.0
    return {"game": game, "history": profile, "history_includes_this_game": bool(this_game_saved and profile)}


def _leaves(obj, nums: list[float], words: set[str]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, str):
        if re.fullmatch(r"[A-Za-z]{5}", obj):
            words.add(obj.lower())
    elif isinstance(obj, dict):
        for v in obj.values():
            _leaves(v, nums, words)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _leaves(v, nums, words)


_NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?%?")


def check_grounding(summary: CoachSummary, facts: dict) -> list[str]:
    """Return a list of problems: figures or capitalised words in the text that are not in the facts."""
    nums: list[float] = []
    words: set[str] = set()
    _leaves(facts, nums, words)
    text = " ".join([summary.headline, summary.next_drill, *summary.went_well, *summary.to_improve])
    problems: list[str] = []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        clean = raw.replace(",", "").replace("%", "").lstrip("+-−")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        if decimals == 0 and value <= 10:
            continue  # ordinals / small counts
        tol = 0.5 * 10 ** -decimals + 1e-9  # a figure may be rounded to the digits the model wrote (5.885 -> 5.89)
        ok = any(abs(abs(x) - value) <= tol or abs(abs(x) * 100 - value) <= tol for x in nums)
        if not ok and f"figure '{raw}'" not in problems:
            problems.append(f"figure '{raw}'")
    for tok in dict.fromkeys(re.findall(r"\b[A-Z]{5}\b", text)):
        if tok.lower() not in words:
            problems.append(f"word '{tok}'")
    return problems


# ---------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------
def _user_prompt(facts: dict) -> str:
    return "Facts (JSON):\n```json\n" + json.dumps(facts, indent=1) + "\n```\nWrite the coaching summary."


def generate_summary(client, facts: dict, model: str | None = None, max_attempts: int = 2) -> SummaryResult:
    model = model or DEFAULT_MODEL
    messages: list[dict] = [{"role": "user", "content": _user_prompt(facts)}]
    res = SummaryResult(summary=None)
    usage = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(1, max_attempts + 1):
        resp = client.messages.create(
            model=model, max_tokens=16000, system=SUMMARY_SYSTEM, messages=messages,
            output_config={"format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}},
        )
        res.attempts = attempt
        u = getattr(resp, "usage", None)
        if u is not None:
            usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            res.error = "The model declined to write this summary."
            break
        if resp.stop_reason == "max_tokens":
            res.error = "The summary was cut off. Try again."
            break
        raw = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        try:
            s = CoachSummary(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError, TypeError):
            res.error = "The model's reply didn't match the summary format."
            break
        s.went_well, s.to_improve = [x.strip() for x in s.went_well if x.strip()][:3], [x.strip() for x in s.to_improve if x.strip()][:3]
        res.summary, res.error = s, None
        res.problems = check_grounding(s, facts)
        if not res.problems:
            res.verified = True
            break
        if attempt < max_attempts:  # ask once for a grounded rewrite
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "These figures or words are not in the facts: " + "; ".join(res.problems) +
                 ". Rewrite the summary using only figures and words that appear in the facts, and compute nothing new."},
            ]
    res.usage = usage
    return res
