from streamlit.testing.v1 import AppTest

from pathlib import Path

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6


def test_add_guess_updates_state():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.text_input(key="word_in").set_value("soare").run()
    at.button[[b.label for b in at.button].index("➕ Add guess")].click().run()
    assert not at.exception, at.exception
    assert at.session_state.history == [("soare", "-----")]


def test_invalid_word_shows_warning():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.text_input(key="word_in").set_value("zzzzz").run()
    at.button[[b.label for b in at.button].index("➕ Add guess")].click().run()
    assert at.warning


def test_coach_tab_loads_example_and_pasted_game():
    at = AppTest.from_file(APP, default_timeout=180).run()
    labels = [b.label for b in at.button]
    at.button[labels.index("Try: A rough game")].click().run()
    assert not at.exception, at.exception
    assert at.session_state.coach_game and at.session_state.coach_game[-1][1] == "GGGGG"
    at.text_area(key="coach_text").set_value("soare -----\nclint --Y-Y\nditty -GGGG\nwitty GGGGG").run()
    at.button[[b.label for b in at.button].index("Analyse pasted game")].click().run()
    assert not at.exception, at.exception
    assert len(at.session_state.coach_game) == 4


def test_coach_rejects_contradictory_paste():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.text_area(key="coach_text").set_value("soare GGGGG\ncrane -----").run()
    at.button[[b.label for b in at.button].index("Analyse pasted game")].click().run()
    assert not at.exception, at.exception
    assert at.error  # "Can't grade this game"
