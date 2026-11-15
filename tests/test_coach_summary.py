import copy
import json
from types import SimpleNamespace as NS

import pytest

from src.coach_summary import SUMMARY_SCHEMA, SUMMARY_SYSTEM, CoachSummary, build_facts, check_grounding, generate_summary
from src.coach_tools import CoachTools
from src.feedback import decode, feedback
from src.memory import PlayerMemory
from src.solvers.entropy import EntropySolver

ROUGH = [(w, decode(feedback(w, "joker"))) for w in ["funny", "bacon", "stove", "poker", "joker"]]


@pytest.fixture(scope="module")
def solver():
    return EntropySolver()


@pytest.fixture(scope="module")
def facts(solver):
    return build_facts(CoachTools(solver, ROUGH).get_game_report(), None)


def S(**kw):
    base = dict(headline="A rough start, but you recovered.", went_well=["You solved it."], to_improve=["Open better."], next_drill="Try SOARE.")
    base.update(kw)
    return CoachSummary(**base)


def test_facts_precompute_bits_lost(facts):
    t0 = facts["game"]["turns"][0]
    assert t0["bits_lost"] == pytest.approx(t0["best_bits"] - t0["expected_bits"], abs=1e-3) and t0["bits_lost"] > 2
    assert facts["game"]["turns"][-1]["bits_lost"] == 0.0 and facts["history_includes_this_game"] is False


@pytest.mark.parametrize("text,ok", [
    ("FUNNY gave 3.33 expected bits against 5.89 for SOARE.", True),
    ("Your skill was 71%.", True),                      # 0.71 in the facts, written as a percentage
    ("You lost 2.55 bits on guess 1.", True),          # precomputed bits_lost
    ("You needed 5 guesses; the solver needs 4.", True),
    ("FUNNY gave 4.5 expected bits.", False),          # invented figure
    ("You lost 12.34 bits overall.", False),           # a computed sum
    ("Try CRANE instead.", False),                     # word that isn't in the facts
    ("About 2,309 answers were possible at the start.", True),  # thousands separator
])
def test_grounding_check(facts, text, ok):
    problems = check_grounding(S(headline=text), facts)
    assert (problems == []) == ok, problems


def test_grounding_uses_history_figures(solver, tmp_path):
    from src.coach import analyze

    mem = PlayerMemory(tmp_path / "h.json")
    mem.save_game(analyze(solver, ROUGH))
    f = build_facts(CoachTools(solver, ROUGH).get_game_report(), mem.profile(), this_game_saved=True)
    assert f["history_includes_this_game"] is True
    avg = f["history"]["avg_skill"]
    assert check_grounding(S(went_well=[f"Your average skill across saved games is {avg}."]), f) == []


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append({**kw, "messages": copy.deepcopy(kw["messages"])})
        r = self.replies.pop(0)
        return r


def reply(summary=None, stop="end_turn", raw=None):
    text = raw if raw is not None else json.dumps(summary)
    return NS(content=[NS(type="text", text=text)], stop_reason=stop, usage=NS(input_tokens=100, output_tokens=50))


GOOD = dict(headline="Weak opener, but you solved it.", went_well=["You solved it in 5 guesses."],
            to_improve=["FUNNY gave 3.33 bits; SOARE gives 5.89."], next_drill="Open with SOARE next game.")
BAD = dict(GOOD, to_improve=["FUNNY gave 9.99 bits."])


def test_verified_first_try_and_request_shape(facts):
    c = FakeClient([reply(GOOD)])
    r = generate_summary(c, facts, model="claude-opus-5-5")
    assert r.verified and r.attempts == 1 and r.problems == [] and r.error is None
    assert r.summary.headline.startswith("Weak opener") and r.usage == {"input_tokens": 100, "output_tokens": 50}
    kw = c.calls[0]
    assert kw["system"] == SUMMARY_SYSTEM and kw["output_config"]["format"] == {"type": "json_schema", "schema": SUMMARY_SCHEMA}
    assert "funny" in kw["messages"][0]["content"] and "bits_lost" in kw["messages"][0]["content"]
    for banned in ("temperature", "top_p", "top_k", "tools", "tool_choice"):
        assert banned not in kw
    assert SUMMARY_SCHEMA["additionalProperties"] is False and set(SUMMARY_SCHEMA["required"]) == set(SUMMARY_SCHEMA["properties"])


def test_ungrounded_figure_triggers_one_rewrite(facts):
    c = FakeClient([reply(BAD), reply(GOOD)])
    r = generate_summary(c, facts)
    assert r.verified and r.attempts == 2 and r.problems == []
    retry = c.calls[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"] and "9.99" in retry[2]["content"]
    assert json.loads(retry[1]["content"])["to_improve"] == BAD["to_improve"]  # the rejected draft is shown back
    assert r.usage["input_tokens"] == 200


def test_still_ungrounded_after_retry_is_reported_not_hidden(facts):
    r = generate_summary(FakeClient([reply(BAD), reply(BAD)]), facts)
    assert r.summary is not None and not r.verified and r.attempts == 2 and any("9.99" in p for p in r.problems)


def test_refusal_truncation_and_bad_json(facts):
    r = generate_summary(FakeClient([reply(stop="refusal", raw="")]), facts)
    assert r.summary is None and "declined" in r.error
    r = generate_summary(FakeClient([reply(GOOD, stop="max_tokens")]), facts)
    assert r.summary is None and "cut off" in r.error
    r = generate_summary(FakeClient([reply(raw="not json")]), facts)
    assert r.summary is None and "format" in r.error
    r = generate_summary(FakeClient([reply({"headline": "x"})]), facts)  # missing required fields
    assert r.summary is None and "format" in r.error


def test_lists_are_trimmed_to_three_and_blank_dropped(facts):
    many = dict(GOOD, went_well=["a", " ", "b", "c", "d"])
    r = generate_summary(FakeClient([reply(many)]), facts)
    assert r.summary.went_well == ["a", "b", "c"]
