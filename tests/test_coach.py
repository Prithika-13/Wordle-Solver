import pytest

from src.coach import analyze, parse_game, parse_pattern
from src.solvers.entropy import EntropySolver


@pytest.fixture(scope="module")
def solver():
    return EntropySolver()


GAME = [("soare", "-----"), ("clint", "--Y-Y"), ("ditty", "-GGGG"), ("aback", "-----"), ("witty", "GGGGG")]


def test_parse_pattern_formats():
    assert parse_pattern("GY--G") == "GY--G"
    assert parse_pattern("🟩🟨⬛⬛🟩") == "GY--G"
    assert parse_pattern("2 1 0 0 2") == "GY--G"
    with pytest.raises(ValueError):
        parse_pattern("GGG")


def test_parse_game():
    assert parse_game("soare -----\n\nclint --Y-Y") == [("soare", "-----"), ("clint", "--Y-Y")]
    with pytest.raises(ValueError):
        parse_game("soare")


def test_best_opener_is_full_marks(solver):
    rep = analyze(solver, [("soare", "-----")])
    t = rep["turns"][0]
    assert t["efficiency"] == pytest.approx(1.0)
    assert t["rating"] == "Excellent"
    assert t["n_before"] == 2309 and t["percentile"] > 99


def test_weak_opener_is_penalised(solver):
    t = analyze(solver, [("fuzzy", "-----")])["turns"][0]
    assert t["efficiency"] < 0.65 and t["rating"] == "Weak"
    assert t["best_word"] == "soare" and t["note"]


def test_luck_is_actual_minus_expected_and_bits_add_up(solver):
    rep = analyze(solver, GAME)
    assert rep["solved"] and rep["n_guesses"] == 5
    for t in rep["turns"]:
        assert t["luck"] == pytest.approx(t["actual_bits"] - t["expected_bits"])
    import math
    total_actual = sum(t["actual_bits"] for t in rep["turns"])
    assert total_actual == pytest.approx(math.log2(2309))  # from 2309 words down to 1


def test_solver_replay_and_verdict(solver):
    rep = analyze(solver, GAME)
    assert rep["solver_game"][-1][1] == "GGGGG"
    assert "entropy solver" in rep["verdict"].lower()
    assert 0 <= rep["skill"] <= 1


def test_contradictory_feedback_rejected(solver):
    with pytest.raises(ValueError):
        analyze(solver, [("soare", "GGGGG"), ("crane", "-----")])
    with pytest.raises(ValueError):
        analyze(solver, [("zzzzz", "-----")])


def test_in_progress_game_has_no_replay(solver):
    rep = analyze(solver, [("crane", "--Y--")])
    assert not rep["solved"] and rep["solver_game"] is None and rep["remaining"] > 1
