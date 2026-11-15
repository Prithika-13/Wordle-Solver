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


def _fake_llm_client():
    """A scripted stand-in for the Anthropic client: one tool call, then a final answer."""
    from types import SimpleNamespace as NS

    script = [
        NS(content=[NS(type="tool_use", id="t1", name="get_game_report", input={})], stop_reason="tool_use", usage=NS(input_tokens=1, output_tokens=1)),
        NS(content=[NS(type="text", text="Your opener scored well; the unlucky colours cost you bits.")], stop_reason="end_turn",
           usage=NS(input_tokens=1, output_tokens=1)),
    ]
    return NS(beta=NS(messages=NS(create=lambda **kw: script.pop(0))))


def test_ask_coach_without_key_shows_setup_not_chat(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert any("ANTHROPIC_API_KEY" in i.value for i in at.info)
    assert len(at.chat_input) == 0


def test_ask_coach_with_key_runs_tool_loop_and_shows_tool_calls(monkeypatch):
    import src.coach_agent as ca

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ca, "make_client", lambda key=None: _fake_llm_client())
    at = AppTest.from_file(APP, default_timeout=180).run()
    labels = [b.label for b in at.button]
    at.button[labels.index("Try: Smart play, unlucky")].click().run()
    assert not at.exception, at.exception
    at.chat_input[0].set_value("Was I unlucky?").run()
    assert not at.exception, at.exception
    chat = at.session_state.coach_chat
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert "unlucky" in chat[1]["content"] and chat[1]["tools"][0]["name"] == "get_game_report"
    assert chat[1]["tools"][0]["output"]["guesses_used"] == 5  # real tool output, computed locally


def test_ask_coach_api_error_is_shown_not_raised(monkeypatch):
    import src.coach_agent as ca

    def boom(**kw):
        raise RuntimeError("network down")

    from types import SimpleNamespace as NS

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ca, "make_client", lambda key=None: NS(beta=NS(messages=NS(create=boom))))
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception, at.exception
    assert at.error and "RuntimeError" in at.error[0].value
    assert at.session_state.coach_chat == []
