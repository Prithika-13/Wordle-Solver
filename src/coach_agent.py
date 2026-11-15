"""'Ask the coach': a Claude tool-use agent that answers questions about a Wordle game.

The LLM never computes anything. It decides which tool to call (src/coach_tools.py), reads the exact
results and explains them in plain language. Every call is recorded so the UI can show what the coach looked at.

Requires the `anthropic` package and credentials (ANTHROPIC_API_KEY, or an `ant auth login` profile).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.coach_tools import TOOL_SCHEMAS, CoachTools

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | Path | None = None) -> bool:
    """Load KEY=value pairs from a .env file (default: <repo>/.env) into the environment.

    Variables that are already set win over the file, so `export ANTHROPIC_API_KEY=...` still takes priority.
    Returns True if a file was read. Does nothing if python-dotenv isn't installed or WORDLE_NO_DOTENV is set.
    """
    if os.environ.get("WORDLE_NO_DOTENV"):
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(Path(path) if path else ROOT / ".env", override=False))


load_env()  # before DEFAULT_MODEL, so COACH_MODEL can be set in .env too

DEFAULT_MODEL = os.environ.get("COACH_MODEL", "claude-opus-5-5")
# Server-side fallback if a safety classifier declines a request (opt-in per the API docs for these models).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-fable-5-2", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are the coach inside a Wordle solver app. The player's game and the solver's analysis are available \
only through your tools. Never recall or estimate numbers yourself: call a tool and quote its result.

Concepts the app uses:
- bits: information. A guess that splits the remaining answers more evenly reveals more bits on average ("expected bits").
- skill (efficiency) = your guess's expected bits / the best available guess's expected bits. It judges the choice.
- luck = bits actually revealed - expected bits. It judges the outcome: positive means the colours were kind.
- "could be the answer" means the guess was still a possible answer, so it could win right away.

How to answer:
- For questions about the player's own guesses, call get_game_report first. Use score_guess to compare alternatives, \
best_guesses for 'what should I play', get_player_history for habits, progress and patterns across saved games, word_difficulty for 'why is this word hard', solve_with_entropy_solver to show \
how the solver plays a word.
- Be concise: lead with the answer, usually under 150 words. Quote specific numbers from the tools and write words in CAPITALS.
- Separate skill from luck: never blame a good choice for an unlucky outcome or praise luck as skill.
- The grader assumes every remaining word is equally likely, so it cannot judge words by familiarity. Say so when it matters.
- History (get_player_history) is computed from games the player saved. Always say how many games a pattern is based on, \
and don't call something a habit or trend from fewer than 3 games. If no games are saved, say so instead of guessing.
- If there are no guesses yet, say so and offer opener advice from best_guesses(after_turn=0).
- If a tool returns an error, fix the input and retry once, otherwise tell the player what failed.
- You only coach Wordle and this game. For anything else, say so briefly."""


@dataclass
class AgentResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "input", "output", "is_error"}
    stop_reason: str | None = None
    steps: int = 0
    usage: dict = field(default_factory=dict)


def make_client(api_key: str | None = None):
    """Anthropic client. With no key it resolves ANTHROPIC_API_KEY / a logged-in profile from the environment."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _request_kwargs(model: str, messages: list) -> dict:
    kwargs: dict[str, Any] = dict(model=model, max_tokens=16000, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, messages=messages)
    if model in FALLBACK_MODELS:
        kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
    return kwargs


def _text_of(content) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def ask_coach(client, tools: CoachTools, question: str, history: list[dict] | None = None, model: str | None = None,
              max_steps: int = 8) -> AgentResult:
    """Answer one question. `history` is prior turns as plain [{"role", "content": str}] (final answers only)."""
    model = model or DEFAULT_MODEL
    messages: list = [dict(m) for m in (history or [])] + [{"role": "user", "content": question}]
    result = AgentResult(text="")
    usage = {"input_tokens": 0, "output_tokens": 0}

    for step in range(1, max_steps + 1):
        resp = client.beta.messages.create(**_request_kwargs(model, messages))
        result.steps = step
        result.stop_reason = resp.stop_reason
        u = getattr(resp, "usage", None)
        if u is not None:
            usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0

        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            result.text = "The model declined to answer this" + (f" ({cat})" if cat else "") + ". Try rephrasing your question about the game."
            break
        if resp.stop_reason == "pause_turn":  # server asked to be resumed; no server tools are used here, but handle it
            messages.append({"role": "assistant", "content": resp.content})
            continue
        if resp.stop_reason != "tool_use":
            result.text = _text_of(resp.content) or "(no answer)"
            if resp.stop_reason == "max_tokens":
                result.text += "\n\n_(the answer was cut off)_"
            break

        # Run every tool the model asked for and return ALL results in a single user message.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            output, is_error = tools.run(block.name, dict(block.input or {}))
            result.tool_calls.append({"name": block.name, "input": dict(block.input or {}), "output": json.loads(output), "is_error": is_error})
            item = {"type": "tool_result", "tool_use_id": block.id, "content": output}
            if is_error:
                item["is_error"] = True
            tool_results.append(item)
        messages.append({"role": "user", "content": tool_results})
    else:
        result.text = ("I looked at the game but ran out of steps before finishing. Try a more specific question.")
        result.stop_reason = "max_steps"

    result.usage = usage
    return result
