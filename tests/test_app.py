from streamlit.testing.v1 import AppTest

from pathlib import Path

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5


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
