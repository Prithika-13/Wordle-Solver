from streamlit.testing.v1 import AppTest

from pathlib import Path

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 7


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


# ----------------------------------------------------------------------------------------------
# memory + AI summary flows
# ----------------------------------------------------------------------------------------------
def _game_text(words, answer):
    from src.feedback import decode, feedback

    return "\n".join(f"{w} {decode(feedback(w, answer))}" for w in words)


PASTED = _game_text(["funny", "bacon", "stove", "poker", "joker"], "joker")


def _paste_and_analyse(at, text=PASTED):
    at.text_area(key="coach_text").set_value(text).run()
    at.button[[b.label for b in at.button].index("Analyse pasted game")].click().run()
    assert not at.exception, at.exception


def _btn(at, label):
    return at.button[[b.label for b in at.button].index(label)]


def test_save_game_to_memory_and_history_tab(monkeypatch, tmp_path):
    import json

    mem = tmp_path / "h.json"
    monkeypatch.setenv("WORDLE_MEMORY_PATH", str(mem))
    at = AppTest.from_file(APP, default_timeout=180).run()
    _paste_and_analyse(at)
    save = _btn(at, "💾 Save to my history")
    assert not save.disabled
    save.click().run()
    assert not at.exception, at.exception
    games = json.loads(mem.read_text())["games"]
    assert len(games) == 1 and games[0]["answer"] == "joker" and games[0]["source"] == "pasted" and games[0]["opener"] == "funny"
    assert _btn(at, "💾 Save to my history").disabled  # same game can't be saved twice
    assert any("Games saved" == m.label and m.value == "1" for m in at.metric)  # the History tab sees it


def test_examples_and_unfinished_games_cannot_be_saved(monkeypatch, tmp_path):
    monkeypatch.setenv("WORDLE_MEMORY_PATH", str(tmp_path / "h.json"))
    at = AppTest.from_file(APP, default_timeout=180).run()
    _btn(at, "Try: A rough game").click().run()
    assert _btn(at, "💾 Save to my history").disabled
    _paste_and_analyse(at, _game_text(["funny", "bacon"], "joker"))  # unfinished
    assert _btn(at, "💾 Save to my history").disabled
    assert not (tmp_path / "h.json").exists()


def _summary_client(payload):
    import json
    from types import SimpleNamespace as NS

    resp = NS(content=[NS(type="text", text=json.dumps(payload))], stop_reason="end_turn", usage=NS(input_tokens=10, output_tokens=7))
    return NS(messages=NS(create=lambda **kw: resp), beta=NS(messages=NS(create=lambda **kw: resp)))


GOOD_SUMMARY = dict(headline="A weak opener cost you, but you solved it.", went_well=["POKER turned a bad start around."],
                    to_improve=["FUNNY gave 3.33 bits while SOARE gives 5.89."], next_drill="Open with SOARE in your next game.")


def test_ai_summary_is_written_and_verified(monkeypatch, tmp_path):
    import src.coach_agent as ca

    monkeypatch.setenv("WORDLE_MEMORY_PATH", str(tmp_path / "h.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ca, "make_client", lambda key=None: _summary_client(GOOD_SUMMARY))
    at = AppTest.from_file(APP, default_timeout=180).run()
    _btn(at, "Try: A rough game").click().run()
    _btn(at, "✍️ Write AI summary").click().run()
    assert not at.exception, at.exception
    res = next(iter(at.session_state.coach_summaries.values()))
    assert res.verified and res.summary.headline.startswith("A weak opener")
    assert any("checked against the report" in c.value for c in at.caption)


def test_ai_summary_flags_invented_figure(monkeypatch, tmp_path):
    import src.coach_agent as ca

    bad = dict(GOOD_SUMMARY, to_improve=["FUNNY gave 9.99 bits."])
    monkeypatch.setenv("WORDLE_MEMORY_PATH", str(tmp_path / "h.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ca, "make_client", lambda key=None: _summary_client(bad))
    at = AppTest.from_file(APP, default_timeout=180).run()
    _btn(at, "Try: A rough game").click().run()
    _btn(at, "✍️ Write AI summary").click().run()
    assert not at.exception, at.exception
    assert any("Could not verify" in w.value and "9.99" in w.value for w in at.warning)


def test_ai_summary_disabled_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("WORDLE_MEMORY_PATH", str(tmp_path / "h.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    at = AppTest.from_file(APP, default_timeout=180).run()
    _btn(at, "Try: A rough game").click().run()
    assert _btn(at, "✍️ Write AI summary").disabled


def test_history_delete_requires_confirmation(monkeypatch, tmp_path):
    mem = tmp_path / "h.json"
    monkeypatch.setenv("WORDLE_MEMORY_PATH", str(mem))
    at = AppTest.from_file(APP, default_timeout=180).run()
    _paste_and_analyse(at)
    _btn(at, "💾 Save to my history").click().run()
    assert mem.exists()
    assert _btn(at, "🗑️ Delete my history").disabled
    at.checkbox(key="confirm_clear").check().run()
    _btn(at, "🗑️ Delete my history").click().run()
    assert not at.exception, at.exception
    assert not mem.exists()
