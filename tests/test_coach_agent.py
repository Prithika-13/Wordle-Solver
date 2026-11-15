"""Agent loop tests with a scripted fake client (no network, no API key)."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from src import coach_agent
from src.coach_agent import FALLBACK_BETA, SYSTEM_PROMPT, ask_coach
from src.coach_tools import TOOL_SCHEMAS, CoachTools
from src.solvers.entropy import EntropySolver

GAME = [("fuzzy", "-----"), ("bacon", "-Y---")]


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def resp(content, stop="end_turn", **extra):
    return NS(content=content, stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5), **extra)


class FakeClient:
    def __init__(self, scripted):
        self.scripted, self.calls = list(scripted), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append({**kw, "messages": copy.deepcopy(kw["messages"])})  # snapshot what was actually sent
        return self.scripted.pop(0)


@pytest.fixture(scope="module")
def tools():
    return CoachTools(EntropySolver(), GAME)


def test_tool_round_trip_message_format(tools):
    c = FakeClient([
        resp([text("Let me check."), tool_use("t1", "get_game_report"), tool_use("t2", "best_guesses", after_turn=0, k=2)], "tool_use"),
        resp([text("Your opener FUZZY was weak; SOARE is best.")]),
    ])
    r = ask_coach(c, tools, "Why was my first guess weak?")
    assert r.text.startswith("Your opener") and r.steps == 2 and [t["name"] for t in r.tool_calls] == ["get_game_report", "best_guesses"]
    assert not any(t["is_error"] for t in r.tool_calls) and r.usage == {"input_tokens": 20, "output_tokens": 10}
    assert len(c.calls[0]["messages"]) == 1  # first request: just the question
    second = c.calls[1]["messages"]
    assert len(second) == 3 and second[0] == {"role": "user", "content": "Why was my first guess weak?"}
    assert second[1]["role"] == "assistant"  # the assistant turn with the tool_use blocks is replayed
    results = second[2]["content"]  # ALL tool results come back in ONE user message
    assert second[2]["role"] == "user" and [x["tool_use_id"] for x in results] == ["t1", "t2"]
    assert all(x["type"] == "tool_result" for x in results)
    assert json.loads(results[0]["content"])["turns"][0]["word"] == "fuzzy"


def test_request_shape_fallbacks_and_tools(tools):
    c = FakeClient([resp([text("ok")])])
    ask_coach(c, tools, "hi", model="claude-opus-5-5")
    kw = c.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["tools"] == TOOL_SCHEMAS and kw["system"] == SYSTEM_PROMPT
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default" and kw["max_tokens"] >= 4096
    for banned in ("temperature", "top_p", "top_k", "tool_choice", "thinking"):  # not valid / not needed on this model
        assert banned not in kw
    c2 = FakeClient([resp([text("ok")])])
    ask_coach(c2, tools, "hi", model="claude-haiku-5-5")
    assert "fallbacks" not in c2.calls[0] and "betas" not in c2.calls[0]


def test_history_is_replayed_as_plain_text(tools):
    c = FakeClient([resp([text("fine")])])
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    ask_coach(c, tools, "q2", history=hist)
    assert c.calls[0]["messages"] == hist + [{"role": "user", "content": "q2"}]


def test_bad_tool_input_is_reported_with_is_error(tools):
    c = FakeClient([resp([tool_use("t1", "score_guess", word="zzzzz")], "tool_use"), resp([text("That word isn't allowed.")])])
    r = ask_coach(c, tools, "score zzzzz")
    assert r.tool_calls[0]["is_error"] is True
    assert c.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_refusal_and_max_tokens_and_step_cap(tools):
    r = ask_coach(FakeClient([resp([], "refusal", stop_details=NS(category="cyber"))]), tools, "x")
    assert "declined" in r.text and "cyber" in r.text
    r = ask_coach(FakeClient([resp([text("partial")], "max_tokens")]), tools, "x")
    assert "cut off" in r.text
    loop = [resp([tool_use(f"t{i}", "best_guesses", after_turn=0)], "tool_use") for i in range(3)]
    r = ask_coach(FakeClient(loop), tools, "x", max_steps=3)
    assert r.stop_reason == "max_steps" and len(r.tool_calls) == 3 and "ran out of steps" in r.text


def test_default_model_is_opus_5_5():
    assert coach_agent.DEFAULT_MODEL in {"claude-opus-5-5"} or "COACH_MODEL" in __import__("os").environ
