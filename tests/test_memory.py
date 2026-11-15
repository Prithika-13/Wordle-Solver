import json

import pytest

from src.coach import analyze
from src.feedback import decode, feedback
from src.memory import PlayerMemory, game_id, is_finished
from src.solvers.entropy import EntropySolver


@pytest.fixture(scope="module")
def solver():
    return EntropySolver()


def play(answer, words):
    return [(w, decode(feedback(w, answer))) for w in words]


def fake_rep(skill, opener="soare", n=4, solved=True, luck=0.0, tag="x"):
    t = dict(turn=1, word=opener, pattern="-----", n_before=2309, n_after=100, expected_bits=5.0, best_bits=5.9, best_word="soare",
             actual_bits=4.0, efficiency=skill, luck=luck, percentile=90.0, could_win=False, rating="Good", note="")
    t2 = dict(t, turn=2, word=f"{tag}ooo", n_before=100)
    return dict(turns=[t, t2], solved=solved, n_guesses=n, skill=skill, luck_bits=luck, solver_game=[("a", "G")] * 4 if solved else None)


def test_save_dedupe_and_roundtrip(solver, tmp_path):
    mem = PlayerMemory(tmp_path / "h.json")
    assert mem.profile() is None and mem.games() == []
    rep = analyze(solver, play("witty", ["soare", "clint", "ditty", "aback", "witty"]))
    r1 = mem.save_game(rep, "played")
    assert r1["saved"] and not r1["duplicate"]
    r2 = mem.save_game(rep, "played")
    assert not r2["saved"] and r2["duplicate"] and r2["id"] == r1["id"]
    g = mem.games()[0]
    assert g["answer"] == "witty" and g["opener"] == "soare" and g["n_guesses"] == 5 and len(g["turns"]) == 5
    assert mem.has(play("witty", ["soare", "clint", "ditty", "aback", "witty"]))
    data = json.loads((tmp_path / "h.json").read_text())
    assert data["version"] == 1 and len(data["games"]) == 1
    assert "api" not in json.dumps(data).lower()  # nothing but game facts


def test_profile_stats_from_real_games(solver, tmp_path):
    mem = PlayerMemory(tmp_path / "h.json")
    mem.save_game(analyze(solver, play("joker", ["funny", "bacon", "stove", "poker", "joker"])))
    mem.save_game(analyze(solver, play("witty", ["funny", "clint", "ditty", "witty"])))
    mem.save_game(analyze(solver, play("crane", ["soare", "pound", "crane"])))
    p = mem.profile()
    assert p["games_saved"] == 3 and p["solved_rate"] == 1.0
    assert p["openers"][0] == {"word": "funny", "games": 2, "avg_efficiency": pytest.approx(p["openers"][0]["avg_efficiency"])}
    assert p["openers"][0]["avg_efficiency"] < 0.65  # FUNNY is a weak opener
    first = p["by_guess_number"][0]
    assert first["guess_number"] == 1 and first["n"] == 3 and first["avg_bits_lost"] > 1
    assert p["weakest_guess_number"] in {1, 2, 3, 4}
    assert p["skill_trend"] is None  # needs 6 games
    assert {"word": "funny", "times": 2} in p["weak_words_played_more_than_once"]
    assert len(p["recent_games"]) == 3 and p["recent_games"][-1]["answer"] == "crane"


def test_trend_needs_six_games_and_detects_direction(tmp_path):
    mem = PlayerMemory(tmp_path / "h.json")
    for i, s in enumerate([0.5, 0.55, 0.6, 0.9, 0.92, 0.95]):
        mem.save_game(fake_rep(s, tag=f"g{i}"))
    tr = mem.profile()["skill_trend"]
    assert tr["direction"] == "improving" and tr["recent_3_avg_skill"] > tr["earlier_avg_skill"]


def test_last_n_and_clear(tmp_path):
    mem = PlayerMemory(tmp_path / "h.json")
    for i in range(4):
        mem.save_game(fake_rep(0.8, tag=f"g{i}"))
    assert len(mem.profile(last_n=2)["recent_games"]) == 2
    assert mem.clear() == 4 and mem.profile() is None and not (tmp_path / "h.json").exists()


def test_corrupt_file_is_preserved_not_overwritten(tmp_path):
    f = tmp_path / "h.json"
    f.write_text("{not json")
    mem = PlayerMemory(f)
    assert mem.games() == []
    assert (tmp_path / "h.corrupt.json").read_text() == "{not json"
    mem.save_game(fake_rep(0.8))
    assert len(mem.games()) == 1


def test_empty_game_rejected_and_finished_rule(tmp_path):
    with pytest.raises(ValueError):
        PlayerMemory(tmp_path / "h.json").save_game(dict(turns=[], solved=False, n_guesses=0, skill=1, luck_bits=0, solver_game=None))
    assert is_finished(dict(solved=True, n_guesses=3)) and is_finished(dict(solved=False, n_guesses=6))
    assert not is_finished(dict(solved=False, n_guesses=3))
    assert game_id([("a", "b")]) == game_id([("a", "b")]) != game_id([("a", "c")])
