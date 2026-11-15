import json

import pandas as pd
import pytest

from src.coach_tools import TOOL_SCHEMAS, CoachTools
from src.solvers.entropy import EntropySolver

GAME = [("soare", "-----"), ("clint", "--Y-Y"), ("ditty", "-GGGG"), ("aback", "-----"), ("witty", "GGGGG")]


@pytest.fixture(scope="module")
def solver():
    return EntropySolver()


@pytest.fixture(scope="module")
def diff():
    return pd.read_csv("results/difficulty.csv")


def call(tools, name, **kw):
    out, err = tools.run(name, kw)
    return json.loads(out), err


def test_schemas_are_valid_and_complete():
    names = [t["name"] for t in TOOL_SCHEMAS]
    assert len(names) == len(set(names)) == 7
    for t in TOOL_SCHEMAS:
        assert t["input_schema"]["type"] == "object" and t["input_schema"]["additionalProperties"] is False
        assert t["description"] and all(r in t["input_schema"]["properties"] for r in t["input_schema"].get("required", []))


def test_game_report_matches_grader(solver, diff):
    t = CoachTools(solver, GAME, diff)
    rep, err = call(t, "get_game_report")
    assert not err and rep["solved"] and rep["guesses_used"] == 5 and len(rep["turns"]) == 5
    assert rep["turns"][0]["word"] == "soare" and rep["turns"][0]["best_word"] == "soare"
    assert rep["entropy_solver_game"][-1]["pattern"] == "GGGGG"


def test_game_report_without_guesses(solver):
    rep, err = call(CoachTools(solver, []), "get_game_report")
    assert "error" in rep and not err  # informative result, not a tool failure


def test_score_guess_opening_and_position(solver):
    t = CoachTools(solver, GAME)
    r, err = call(t, "score_guess", word="fuzzy", after_turn=0)
    assert not err and r["answers_remaining"] == 2309 and r["best_word"] == "soare"
    assert r["efficiency_vs_best"] < 0.65
    r2, _ = call(t, "score_guess", word="clint", after_turn=1)  # position after the first guess
    assert r2["answers_remaining"] == 182 and r2["efficiency_vs_best"] == pytest.approx(1.0)


def test_best_guesses_and_remaining(solver):
    t = CoachTools(solver, GAME)
    b, _ = call(t, "best_guesses", after_turn=0, k=3)
    assert [x["word"] for x in b["best"]][0] == "soare" and len(b["best"]) == 3
    r, _ = call(t, "remaining_answers", after_turn=2, limit=5)
    assert r["answers_remaining"] == 9 and len(r["words"]) == 5 and r["list_truncated"]
    d, _ = call(t, "remaining_answers")  # default position: after all guesses
    assert d["answers_remaining"] == 1 and d["words"] == ["witty"]


def test_solver_and_difficulty_tools(solver, diff):
    t = CoachTools(solver, [], diff)
    s, _ = call(t, "solve_with_entropy_solver", answer="witty")
    assert s["guesses_needed"] == 5 and s["game"][-1]["pattern"] == "GGGGG"
    d, _ = call(t, "word_difficulty", word="watch")
    assert d["look_alike_cluster"] == "_atch" and d["difficulty_rank"] <= 5 and "batch" in d["look_alikes"]


@pytest.mark.parametrize("name,kw", [
    ("score_guess", dict(word="zzzzz")), ("score_guess", dict(word="soare", after_turn=99)),
    ("score_guess", dict(word=123)), ("best_guesses", dict(k=50)), ("remaining_answers", dict(limit=-1)),
    ("solve_with_entropy_solver", dict(answer="soare")),  # valid guess but not an answer
    ("word_difficulty", dict(word="qqqqq")), ("nope", {}), ("score_guess", dict(word="soare", bogus=1)),
])
def test_bad_input_becomes_error_result_never_raises(solver, diff, name, kw):
    out, err = CoachTools(solver, GAME, diff).run(name, kw)
    assert err is True and "error" in json.loads(out)


def test_get_player_history_tool(solver, tmp_path):
    from src.coach import analyze
    from src.memory import PlayerMemory

    t_none = CoachTools(solver, GAME)  # no memory configured
    out, err = call(t_none, "get_player_history")
    assert not err and out["games_saved"] == 0 and "No games" in out["note"]

    mem = PlayerMemory(tmp_path / "h.json")
    t = CoachTools(solver, GAME, memory=mem)
    assert call(t, "get_player_history")[0]["games_saved"] == 0
    mem.save_game(analyze(solver, GAME))
    out, err = call(t, "get_player_history", last_n=1)
    assert not err and out["games_saved"] == 1 and out["openers"][0]["word"] == "soare" and len(out["recent_games"]) == 1
    assert call(t, "get_player_history", last_n=99)[1] is True
