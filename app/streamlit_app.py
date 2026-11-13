"""Wordle Solver Lab - an interactive dashboard for the entropy solver, the RL agents and the analyses.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.feedback import ALL_GREEN, decode, encode
from src.solvers.entropy import EntropySolver, entropy_scores
from src.solvers.heuristic import FrequencyHeuristic

RES = ROOT / "results"
REPO = "https://github.com/Prithika-13/Wordle-Solver"

st.set_page_config(page_title="Wordle Solver Lab", page_icon="🟩", layout="wide")

# ----------------------------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------------------------
GREEN, YELLOW, GRAY = "#538d4e", "#b59f3b", "#3a3a3c"
COLORS = {"G": GREEN, "Y": YELLOW, "-": GRAY}
GROUP_COLORS = {
    "Random": "#8b8d91",
    "Frequency heuristic": "#c9a227",
    "Entropy (candidates only)": "#74a9cf",
    "Entropy (all guesses)": "#2f7fd8",
    "RL blind": "#e0762b",
    "RL assisted": "#3fbf8f",
}
st.markdown(
    f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1250px;}}
.hero h1 {{margin-bottom: 0; letter-spacing: -0.5px;}}
.hero p {{color:#a9a9ad; margin-top: .2rem; font-size: 1.05rem;}}
.tile {{display:inline-flex; align-items:center; justify-content:center; width:2.7rem; height:2.7rem; margin:2px;
       font-weight:800; font-size:1.35rem; color:#fff; border-radius:5px; font-family:'Helvetica Neue',Arial,sans-serif;}}
.tile.empty {{border:2px solid {GRAY}; background:transparent;}}
.tile.sm {{width:1.85rem; height:1.85rem; font-size:.95rem; margin:1.5px; border-radius:4px;}}
.tile.plain {{background:#2a2a2c; border:1px solid #4a4a4d;}}
.board {{line-height:0;}}
.pill {{display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600;
       margin-right:.35rem; color:#fff;}}
.bigword {{font-size:2.0rem; font-weight:800; letter-spacing:4px; margin:.1rem 0 .2rem 0;}}
.muted {{color:#a9a9ad; font-size:.9rem;}}
.chip {{display:inline-block; background:#2a2a2c; border-radius:6px; padding:.1rem .5rem; margin:2px; font-family:monospace;}}
.takeaway {{border-left:4px solid {GREEN}; background:#1d1d1f; padding:.7rem 1rem; border-radius:4px; margin:.5rem 0;}}
</style>
""",
    unsafe_allow_html=True,
)


def tiles(word: str, pattern: str | None = None, size: str = "") -> str:
    """HTML for one row of Wordle tiles. pattern uses G / Y / - ; None => neutral tiles."""
    out = []
    for i, ch in enumerate(word.upper().ljust(5)):
        if pattern is None:
            out.append(f"<span class='tile plain {size}'>{ch.strip()}</span>")
        else:
            out.append(f"<span class='tile {size}' style='background:{COLORS[pattern[i]]}'>{ch}</span>")
    return "".join(out)


def empty_row(size: str = "") -> str:
    return "".join(f"<span class='tile empty {size}'></span>" for _ in range(5))


def board(history: list[tuple[str, str]], rows: int = 6, size: str = "") -> str:
    html = [f"<div>{tiles(w, p, size)}</div>" for w, p in history]
    html += [f"<div>{empty_row(size)}</div>" for _ in range(max(0, rows - len(history)))]
    return f"<div class='board'>{''.join(html)}</div>"


def hbars(rows: list[tuple[str, float, str]], lo: float, hi: float, label_w: str = "11.5rem") -> str:
    """Simple HTML horizontal bar list: rows of (label, value, colour). Robust in any theme."""
    out = []
    for label, val, colr in rows:
        w = max(2.0, 100 * (val - lo) / (hi - lo))
        out.append(
            f"<div style='display:flex;align-items:center;margin:.55rem 0'>"
            f"<div style='width:{label_w};font-size:.9rem'>{label}</div>"
            f"<div style='flex:1;background:#1d1d1f;border-radius:5px'>"
            f"<div style='width:{w:.1f}%;background:{colr};height:1.55rem;border-radius:5px'></div></div>"
            f"<div style='width:3.6rem;text-align:right;font-weight:700'>{val:.2f}</div></div>"
        )
    return "".join(out)


def pill(text: str, color: str) -> str:
    return f"<span class='pill' style='background:{color}'>{text}</span>"


# ----------------------------------------------------------------------------------------------
# cached resources
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading solvers (first run builds the feedback matrix, ~20 s)...")
def load_solvers():
    solvers = {
        "Entropy": EntropySolver(),
        "Frequency heuristic": FrequencyHeuristic(),
    }
    rl_ok = True
    try:
        from src.solvers.rl_agent import RLSolver

        for label, ckpt in [("RL assisted", "rl_assisted_s0.pt"), ("RL blind", "rl_blind_s0.pt")]:
            if (RES / ckpt).exists():
                solvers[label] = RLSolver(RES / ckpt, name=label)
    except Exception as e:  # torch missing etc.
        rl_ok = False
        solvers["_error"] = str(e)  # type: ignore[assignment]
    return solvers, rl_ok


SOLVERS, RL_OK = load_solvers()
ENT: EntropySolver = SOLVERS["Entropy"]
M, ANSWERS, GUESSES = ENT.M, ENT.answers, ENT.guesses
GI = ENT.guess_index
N_ANS = len(ANSWERS)
TOTAL_BITS = float(np.log2(N_ANS))
SOLVER_META = {
    "Entropy": ("Information theory", "#2f7fd8"),
    "RL assisted": ("Machine learning · PPO", GROUP_COLORS["RL assisted"]),
    "RL blind": ("Machine learning · PPO", GROUP_COLORS["RL blind"]),
    "Frequency heuristic": ("Hand-written rule", GROUP_COLORS["Frequency heuristic"]),
}
PLAYERS = [n for n in ["Entropy", "RL assisted", "RL blind", "Frequency heuristic"] if n in SOLVERS]


def candidates_after(history: list[tuple[str, str]]):
    cand = np.arange(N_ANS)
    key = ()
    for w, pat in history:
        g, code = GI[w], encode(pat)
        cand = cand[M[g, cand] == code]
        key += ((g, code),)
    return cand, key


@st.cache_data(show_spinner=False)
def suggestions(history: tuple) -> dict:
    cand, key = candidates_after(list(history))
    n = len(cand)
    out = dict(n=n, words=[ANSWERS[i] for i in cand[:60]], top=[], picks={})
    if n < 2:
        return out
    h = entropy_scores(M, cand)
    is_cand = np.zeros(len(GUESSES), bool)
    is_cand[ENT.ans_as_guess[cand]] = True
    score = h + is_cand / n
    for i in np.argsort(-score)[:5]:
        out["top"].append((GUESSES[i], float(h[i]), bool(is_cand[i])))
    for name in PLAYERS:
        g = SOLVERS[name].choose(cand, key, len(history))
        out["picks"][name] = (GUESSES[g], float(h[g]), bool(is_cand[g]))
    return out


# ----------------------------------------------------------------------------------------------
# results data
# ----------------------------------------------------------------------------------------------
def group_name(s: str) -> str:
    if s.startswith("rl_assisted"):
        return "RL assisted"
    if s.startswith("rl_blind"):
        return "RL blind"
    return {"random": "Random", "frequency": "Frequency heuristic", "entropy-cand": "Entropy (candidates only)",
            "entropy": "Entropy (all guesses)"}.get(s, s)


@st.cache_data(show_spinner=False)
def load_eval() -> pd.DataFrame | None:
    f = RES / "eval_all.csv"
    if not f.exists():
        return None
    ev = pd.read_csv(f)
    ev["group"] = ev.solver.map(group_name)
    return ev


@st.cache_data(show_spinner=False)
def load_logs() -> pd.DataFrame | None:
    frames = []
    for f in sorted(RES.glob("rl_*_log.csv")):
        d = pd.read_csv(f)
        d["run"] = f.name.replace("_log.csv", "")
        d["group"] = "RL assisted" if "assisted" in f.name else "RL blind"
        frames.append(d)
    return pd.concat(frames) if frames else None


@st.cache_data(show_spinner=False)
def load_difficulty():
    f, r = RES / "difficulty.csv", RES / "difficulty_report.json"
    if not (f.exists() and r.exists()):
        return None, None
    return pd.read_csv(f), json.loads(r.read_text())


# ----------------------------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------------------------
st.markdown(
    f"""<div class='hero'><h1>🟩 Wordle Solver Lab</h1>
<p>Information theory vs. reinforcement learning, head to head. {pill('Entropy solver', '#2f7fd8')}
{pill('RL agent (PPO)', GROUP_COLORS['RL assisted'])} {pill('Difficulty analysis', '#e0762b')}
<a href='{REPO}' target='_blank' style='color:#a9a9ad'>source on GitHub ↗</a></p></div>""",
    unsafe_allow_html=True,
)
if not RL_OK:
    st.warning("PyTorch isn't installed, so the RL agents are hidden. Run `pip install torch` to enable them.")

tab_play, tab_watch, tab_results, tab_insights, tab_about = st.tabs(
    ["🎮 Solve a game", "🔎 Watch a word", "📊 Results", "🧠 Why words are hard", "📖 How it works"]
)

# ==============================================================================================
# TAB 1 - solve a real game
# ==============================================================================================
PAT = {"⬛": "-", "🟨": "Y", "🟩": "G"}


def _init_state():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("word_in", "")
    st.session_state.setdefault("msg", None)
    for i in range(5):
        st.session_state.setdefault(f"c{i}", "⬛")


def add_guess():
    w = st.session_state.word_in.strip().lower()
    pat = "".join(PAT[st.session_state.get(f"c{i}") or "⬛"] for i in range(5))
    if len(w) != 5 or w not in GI:
        st.session_state.msg = ("warning", f"“{w or '…'}” isn't a valid Wordle word. Enter a 5-letter word from the allowed list.")
        return
    new = st.session_state.history + [(w, pat)]
    cand, _ = candidates_after(new)
    if len(cand) == 0:
        st.session_state.msg = ("error", "No possible answer matches that feedback. Check the colours you entered.")
        return
    st.session_state.history = new
    st.session_state.word_in = ""
    for i in range(5):
        st.session_state[f"c{i}"] = "⬛"
    st.session_state.msg = None


CYCLE = ["⬛", "🟨", "🟩"]


def cycle_colour(i: int):
    cur = st.session_state.get(f"c{i}") or "⬛"
    st.session_state[f"c{i}"] = CYCLE[(CYCLE.index(cur) + 1) % 3]


def use_word(w: str):
    st.session_state.word_in = w


def undo():
    if st.session_state.history:
        st.session_state.history = st.session_state.history[:-1]
    st.session_state.msg = None


def new_game():
    st.session_state.history = []
    st.session_state.word_in = ""
    st.session_state.msg = None


_init_state()

with tab_play:
    left, right = st.columns([5, 7], gap="large")
    hist = st.session_state.history
    sug = suggestions(tuple(hist))

    with left:
        st.subheader("Your game")
        st.caption("Play a word in the real Wordle (or any Wordle clone), then tell me the colours you got.")
        st.markdown(board(hist), unsafe_allow_html=True)
        st.write("")
        if hist and hist[-1][1] == "GGGGG":
            st.success(f"Solved in {len(hist)} guesses! 🎉")
        elif sug["n"] == 1:
            st.success(f"Only one word left: **{sug['words'][0].upper()}**")
        else:
            st.text_input("Word you guessed", key="word_in", max_chars=5, placeholder="e.g. soare")
            st.caption("Click each square to cycle its colour: gray → yellow → green")
            cols = st.columns(5)
            for i, c in enumerate(cols):
                c.button(st.session_state.get(f"c{i}") or "⬛", key=f"cb{i}", on_click=cycle_colour, args=(i,), width="stretch")
            cur = st.session_state.word_in.strip().lower().ljust(5)[:5]
            cur_pat = "".join(PAT[st.session_state.get(f"c{i}") or "⬛"] for i in range(5))
            st.markdown(f"<div class='board'>{tiles(cur, cur_pat)}</div>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns([2, 1, 1])
            b1.button("➕ Add guess", on_click=add_guess, type="primary", width="stretch")
            b2.button("↩️ Undo", on_click=undo, width="stretch", disabled=not hist)
            b3.button("🔄 New", on_click=new_game, width="stretch")
        if st.session_state.msg:
            getattr(st, st.session_state.msg[0])(st.session_state.msg[1])
        if hist and (hist[-1][1] == "GGGGG" or sug["n"] == 1):
            st.button("🔄 New game", on_click=new_game)

    with right:
        n = sug["n"]
        st.subheader("What should I play next?")
        c1, c2, c3 = st.columns(3)
        c1.metric("Possible answers left", f"{n:,}")
        bits_left = float(np.log2(max(n, 1)))
        c2.metric("Uncertainty left", f"{bits_left:.1f} bits", help=f"Started at {TOTAL_BITS:.1f} bits (log2 of {N_ANS:,} answers).")
        c3.metric("Guess number", f"{len(hist) + 1} of 6")
        st.progress(min(1.0, (TOTAL_BITS - bits_left) / TOTAL_BITS),
                    text=f"Information gained so far: {TOTAL_BITS - bits_left:.1f} of {TOTAL_BITS:.1f} bits")

        if n >= 2:
            picks = sug["picks"]
            for row in range(0, len(PLAYERS), 2):
                cols = st.columns(2)
                for col, name in zip(cols, PLAYERS[row : row + 2]):
                    word, bits, cand_flag = picks[name]
                    kind, colr = SOLVER_META[name]
                    with col, st.container(border=True):
                        st.markdown(f"{pill(kind, colr)} **{name}**", unsafe_allow_html=True)
                        st.markdown(f"<div class='board' style='margin:.35rem 0'>{tiles(word, None)}</div>", unsafe_allow_html=True)
                        st.caption(f"{bits:.2f} bits expected · " + ("could win now ✅" if cand_flag else "probe word 🔍"))
                        st.button("Use this word", key=f"use_{name}", on_click=use_word, args=(word,), width="stretch")
            words = {w for w, _, _ in picks.values()}
            if len(words) == 1:
                st.success(f"All {len(PLAYERS)} solvers agree: **{next(iter(words)).upper()}**")
            else:
                st.info(
                    "The solvers disagree. That's normal: several words carry almost the same information. "
                    "Entropy always plays the highest expected-information word; the RL agents learned their own preferences."
                )
            with st.expander("Top 5 words by expected information (entropy solver)", expanded=False):
                st.dataframe(
                    pd.DataFrame([{"word": w.upper(), "expected bits": round(b, 3), "could be the answer": c}
                                  for w, b, c in sug["top"]]),
                    hide_index=True, width="stretch",
                )
            if n <= 40:
                with st.expander(f"All {n} remaining answers", expanded=n <= 12):
                    st.markdown(" ".join(f"<span class='chip'>{w}</span>" for w in sug["words"]), unsafe_allow_html=True)
        elif n == 0:
            st.error("No word matches that feedback.")

# ==============================================================================================
# TAB 2 - watch solvers play a word
# ==============================================================================================
diff_df, diff_rep = load_difficulty()


def trace(solver, ans_idx: int):
    guesses = solver.play(ans_idx)
    rows, cand = [], np.arange(N_ANS)
    for g in guesses:
        code = int(M[g, ans_idx])
        before = len(cand)
        cand = cand[M[g, cand] == code] if code != ALL_GREEN else np.array([ans_idx])
        rows.append(dict(word=GUESSES[g], pattern=decode(code), left=len(cand), bits=float(np.log2(before / max(len(cand), 1)))))
    solved = rows[-1]["pattern"] == "GGGGG"
    return rows, solved


def set_watch(w: str):
    st.session_state.watch_word = w


with tab_watch:
    st.subheader("Watch the solvers play the same word")
    st.caption("Pick a hidden answer. Every solver plays it from scratch, so you can see where they differ.")
    ans_sorted = sorted(ANSWERS)
    st.session_state.setdefault("watch_word", "witty")
    top = st.columns([3, 1.4, 5])
    top[0].selectbox("Hidden answer", ans_sorted, key="watch_word")
    top[1].write("")
    top[1].button("🎲 Random word", on_click=lambda: set_watch(random.choice(ans_sorted)), width="stretch")
    if diff_df is not None:
        hard = diff_df.sort_values(["difficulty", "word"], ascending=[False, True]).word.head(5).tolist()
        top[2].caption("Try the hardest words")
        hc = top[2].columns(5)
        for c, w in zip(hc, hard):
            c.button(w, key=f"hard_{w}", on_click=set_watch, args=(w,), width="stretch")

    target = st.session_state.watch_word
    ai = ANSWERS.index(target)
    if diff_df is not None:
        row = diff_df[diff_df.word == target].iloc[0]
        rank = int((diff_df.difficulty > row.difficulty).sum()) + 1
        st.markdown(
            f"**{target.upper()}** needs on average **{row.difficulty:.2f}** guesses across solvers "
            f"(difficulty rank **#{rank} of {N_ANS:,}**). "
            + (f"Look-alike cluster: `{row.cluster_pattern}` ({int(row.cluster_size)} words)." if row.cluster_size else "No look-alike cluster.")
        )
    cols = st.columns(len(PLAYERS))
    for col, name in zip(cols, PLAYERS):
        rows, solved = trace(SOLVERS[name], ai)
        kind, colr = SOLVER_META[name]
        with col, st.container(border=True):
            st.markdown(pill(kind, colr), unsafe_allow_html=True)
            st.markdown(f"**{name}** · " + (f"solved in **{len(rows)}**" if solved else "**failed** ❌"))
            st.markdown(board([(r["word"], r["pattern"]) for r in rows], rows=6, size="sm"), unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame([{"#": i + 1, "guess": r["word"].upper(), "left": r["left"], "bits": round(r["bits"], 1)}
                              for i, r in enumerate(rows)]),
                hide_index=True, width="stretch", height=38 + 35 * len(rows),
            )
    st.caption("“left” = answers still possible after that guess · “bits” = information that guess revealed.")

# ==============================================================================================
# TAB 3 - results
# ==============================================================================================
with tab_results:
    ev = load_eval()
    if ev is None:
        st.info("Run `python scripts/compare.py` to generate the comparison results.")
    else:
        st.subheader("How do the solvers compare?")
        split = st.radio("Evaluate on", ["Held-out test answers (461)", "Training answers (1,848)", "All answers (2,309)"],
                         horizontal=True)
        sub = ev if split.startswith("All") else ev[ev.split == ("test" if split.startswith("Held") else "train")]
        per_solver = sub.groupby(["group", "solver"]).agg(
            mean=("score", "mean"), solve=("solved", "mean"), median=("score", "median")).reset_index()
        grp = per_solver.groupby("group").agg(
            mean=("mean", "mean"), std=("mean", "std"), solve=("solve", "mean"), median=("median", "mean"),
            runs=("solver", "count")).reset_index()
        grp["std"] = grp["std"].fillna(0.0)
        order = list(GROUP_COLORS)
        grp = grp.set_index("group")
        grp = grp.reindex([g for g in order if g in grp.index]).rename_axis("group").reset_index()
        ent_mean = float(grp.loc[grp.group == "Entropy (all guesses)", "mean"].iloc[0])

        mcols = st.columns(4)
        for col, g in zip(mcols, ["Entropy (all guesses)", "RL assisted", "RL blind", "Frequency heuristic"]):
            r = grp[grp.group == g]
            if len(r):
                v = float(r["mean"].iloc[0])
                col.metric(g, f"{v:.2f} guesses", None if g.startswith("Entropy (all") else f"{v - ent_mean:+.2f} vs entropy",
                           delta_color="inverse")
        st.caption("Average guesses per game (a failed game counts as 7). Lower is better. RL numbers are the mean of 3 training seeds.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Average guesses** (lower is better)")
            lo, hi = 3.0, float(grp["mean"].max()) + 0.05
            st.markdown(hbars([(r["group"], r["mean"], GROUP_COLORS[r["group"]]) for _, r in grp.iterrows()], lo, hi),
                        unsafe_allow_html=True)
            st.caption(f"Bars start at {lo:.0f} guesses. Failed games count as 7. Solve rates: "
                       + ", ".join(f"{r['group']} {r['solve']:.1%}" for _, r in grp.iterrows() if r['group'] in ('Entropy (all guesses)', 'RL assisted', 'RL blind')))
        with c2:
            st.markdown("**How many guesses does it take?**")
            cat = sub.assign(k=np.where(sub.solved, sub.n_guesses.astype(str), "fail"))
            dist = cat.groupby(["group", "solver", "k"]).size().rename("n").reset_index()
            dist["frac"] = dist.n / dist.groupby(["group", "solver"]).n.transform("sum")
            dist = dist.groupby(["group", "k"]).frac.mean().reset_index()
            dist = dist[dist.group.isin(["Frequency heuristic", "RL blind", "RL assisted", "Entropy (all guesses)"])]
            ch = alt.Chart(dist).mark_bar().encode(
                x=alt.X("k:N", sort=["1", "2", "3", "4", "5", "6", "fail"], title="guesses to solve", axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("group:N", sort=None),
                y=alt.Y("frac:Q", axis=alt.Axis(format="%"), title="share of games"),
                color=alt.Color("group:N", sort=None, scale=alt.Scale(domain=list(GROUP_COLORS), range=list(GROUP_COLORS.values())),
                                legend=None),
                tooltip=["group", "k", alt.Tooltip("frac:Q", format=".1%")],
            )
            st.altair_chart(ch.properties(height=290), width="stretch")
            st.caption("Same colours as the bars on the left (random is left out for readability).")

        logs = load_logs()
        if logs is not None:
            st.markdown("**Did the RL agents actually learn?** Held-out score while training (greedy play):")
            lg = logs.dropna(subset=["test_mean_guesses"])
            line = alt.Chart(lg).mark_line(point=True).encode(
                x=alt.X("episodes:Q", title="training games played"),
                y=alt.Y("test_mean_guesses:Q", title="mean guesses (held-out)", scale=alt.Scale(domain=[3.3, 3.9])),
                color=alt.Color("group:N", scale=alt.Scale(domain=["RL blind", "RL assisted"],
                                range=[GROUP_COLORS["RL blind"], GROUP_COLORS["RL assisted"]]), legend=alt.Legend(orient="top", title=None)),
                detail="run:N", tooltip=["run", "episodes", alt.Tooltip("test_mean_guesses:Q", format=".3f")],
            )
            ref_val = float(ev[(ev.split == "test") & (ev.group == "Entropy (all guesses)")].score.mean())
            rule = alt.Chart(pd.DataFrame({"y": [ref_val]})).mark_rule(strokeDash=[6, 4], color=GROUP_COLORS["Entropy (all guesses)"]).encode(y="y:Q")
            st.altair_chart((line + rule).properties(height=260), width="stretch")
            st.caption(f"Dashed line: entropy solver on the same held-out answers ({ref_val:.2f}).")

        with st.expander("Full table"):
            show = grp.rename(columns={"group": "solver", "mean": "mean guesses", "std": "± across seeds", "solve": "solve rate",
                                       "median": "median guesses"})
            st.dataframe(show.style.format({"mean guesses": "{:.3f}", "± across seeds": "{:.3f}", "solve rate": "{:.1%}",
                                            "median guesses": "{:.1f}"}), hide_index=True, width="stretch")

        g = grp.set_index("group")["mean"]
        if {"RL assisted", "RL blind", "Entropy (all guesses)"} <= set(g.index):
            st.markdown(
                f"<div class='takeaway'><b>Takeaways.</b> The entropy solver is the one to beat ({g['Entropy (all guesses)']:.2f}). "
                f"The RL agent that was given entropy features lands at {g['RL assisted']:.2f}, essentially a tie, so reinforcement "
                f"learning matched the principled method but didn't beat it. Without those features the RL agent plateaus at "
                f"{g['RL blind']:.2f}, the same level as a hand-written letter-frequency rule ({g.get('Frequency heuristic', float('nan')):.2f}).</div>",
                unsafe_allow_html=True,
            )

# ==============================================================================================
# TAB 4 - difficulty analysis
# ==============================================================================================
with tab_insights:
    if diff_df is None:
        st.info("Run `python scripts/difficulty.py` to generate the difficulty analysis.")
    else:
        rep = diff_rep
        r2 = rep["cv_r2"]
        st.subheader("Why are some words so hard?")
        st.markdown(
            "Difficulty here means the **average number of guesses** the strong solvers need for a word "
            "(entropy solver + 3 trained RL agents; failures count as 7). I then trained simple, interpretable "
            "models to find out *which properties of a word* predict it."
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Words analysed", f"{rep['n_words']:,}")
        k2.metric("Average difficulty", f"{rep['mean_difficulty']:.2f} guesses")
        k3.metric("Variance explained (CV R²)", f"{r2['linear']:.0%}", help="5-fold cross-validated R² of a linear model on 8 word features.")
        k4.metric("Best model (random forest)", f"{r2['random forest']:.0%}")
        st.markdown(
            f"<div class='takeaway'><b>Honest read.</b> Word features explain only about <b>{r2['linear']:.0%}</b> of the difference in "
            f"difficulty, and even a random forest reaches just {r2['random forest']:.0%}. The rest depends on the <i>specific</i> "
            f"words that happen to survive each guess, which simple word features can't capture.</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2, gap="large")
        short = {"cluster_size": "Look-alike cluster size", "opener_bucket": "Crowded after best opener",
                 "dup_letters": "Repeated letters", "mean_letter_freq": "Letter commonness",
                 "min_letter_freq": "Has a rare letter", "pos_score": "Positional frequency", "n_vowels": "Vowel count",
                 "n_neighbors": "One-letter neighbours"}
        coef = pd.DataFrame(rep["coefficients"])
        coef["name"] = coef.feature.map(short)
        # rarity features are stored as frequency (higher = more common); flip sign of the two commonness features for reading
        coef["effect"] = np.where(coef.feature.isin(["mean_letter_freq", "min_letter_freq"]), -coef.coef, coef.coef)
        coef["lo2"] = np.where(coef.feature.isin(["mean_letter_freq", "min_letter_freq"]), -coef.hi, coef.lo)
        coef["hi2"] = np.where(coef.feature.isin(["mean_letter_freq", "min_letter_freq"]), -coef.lo, coef.hi)
        coef["name"] = coef["name"].replace({"Letter commonness": "Uncommon letters (avg)", "Has a rare letter": "Has a rare letter"})
        coef["sig"] = np.where((coef.lo2 > 0) | (coef.hi2 < 0), "clear effect", "unclear")
        with c1:
            st.markdown("**What makes a word harder?**")
            order_names = list(coef.sort_values("effect", ascending=False)["name"])
            base = alt.Chart(coef).encode(y=alt.Y("name:N", sort=order_names, title=None, axis=alt.Axis(labelLimit=220)))
            bars = base.mark_bar().encode(
                x=alt.X("effect:Q", title="extra guesses per +1 std dev of the feature"),
                color=alt.Color("sig:N", scale=alt.Scale(domain=["clear effect", "unclear"], range=["#e0762b", "#6b6b70"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["name", alt.Tooltip("effect:Q", format="+.3f"), alt.Tooltip("lo2:Q", format="+.3f", title="95% CI low"),
                         alt.Tooltip("hi2:Q", format="+.3f", title="95% CI high")],
            )
            ci = alt.Chart(coef).mark_rule(color="#f2f2f2", strokeWidth=2).encode(
                y=alt.Y("name:N", sort=order_names, bandPosition=0.5), x="lo2:Q", x2="hi2:Q")
            st.altair_chart((bars + ci).properties(height=300), width="stretch")
            st.caption("Bars show the effect of each feature with a 95% bootstrap interval. Grey = can't rule out zero.")
        with c2:
            st.markdown("**Crowded neighbourhoods are harder**")
            cc = pd.DataFrame(rep["cluster_curve"])
            st.markdown(
                hbars([(f"{r['bin']} look-alikes<br><span class='muted'>{r['words']:,} words</span>" if r["bin"] != "none"
                        else f"no cluster<br><span class='muted'>{r['words']:,} words</span>", r["mean_difficulty"], "#e0762b")
                       for _, r in cc.iterrows()], 3.0, float(cc.mean_difficulty.max()) + 0.05, label_w="9rem"),
                unsafe_allow_html=True,
            )
            st.caption("Average guesses by size of the look-alike cluster (words sharing 4 letters in place). Bars start at 3 guesses.")
            st.caption("Clusters like batch/catch/hatch/latch/watch force you to guess letter by letter.")

        st.markdown("### The 20 hardest words")
        hd = pd.DataFrame(rep["hardest"])
        hd.insert(0, "rank", range(1, len(hd) + 1))
        hd["word"] = hd.word.str.upper()
        hd["look-alikes"] = hd.cluster_members.str.split(", ").str[:6].str.join(", ")
        hd["cluster"] = np.where(hd.cluster_size > 0, hd.cluster_pattern.str.upper() + " (" + hd.cluster_size.astype(str) + ")", "none")
        st.dataframe(
            hd[["rank", "word", "difficulty", "cluster", "look-alikes", "dup_letters", "min_letter_freq"]].rename(
                columns={"difficulty": "avg guesses", "dup_letters": "repeats", "min_letter_freq": "rarest letter freq"}),
            hide_index=True, width="stretch", height=38 + 35 * 20,
            column_config={
                "avg guesses": st.column_config.ProgressColumn("avg guesses", min_value=3.0, max_value=6.0, format="%.2f"),
                "rarest letter freq": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption("Two patterns: (1) many look-alikes, e.g. WATCH/BATCH, WIGHT, BOUND/WOUND; (2) a rare letter or repeated letters, e.g. BOXER, JOKER, POPPY.")

        st.markdown("### Look up any word")
        lw = st.selectbox("Pick a word", sorted(ANSWERS), index=sorted(ANSWERS).index("joker"), key="lookup")
        r = diff_df[diff_df.word == lw].iloc[0]
        rank = int((diff_df.difficulty > r.difficulty).sum()) + 1
        a, b, c = st.columns([1.2, 1.2, 2.6])
        a.metric("Avg guesses", f"{r.difficulty:.2f}", f"{r.difficulty - rep['mean_difficulty']:+.2f} vs average", delta_color="inverse")
        b.metric("Difficulty rank", f"#{rank} of {N_ANS:,}", help="1 = hardest word")
        with c:
            st.markdown(tiles(lw, None), unsafe_allow_html=True)
            if r.cluster_size:
                members = [m for m in str(r.cluster_members).split(", ") if m][:12]
                st.markdown(f"Look-alikes (`{r.cluster_pattern}`): " + " ".join(f"<span class='chip'>{m}</span>" for m in members),
                            unsafe_allow_html=True)
            else:
                st.markdown("No look-alike cluster.")
        feat = pd.DataFrame({
            "feature": ["Look-alike cluster size", "Repeated letters", "Rarest letter appears in % of answers",
                        "Words sharing its colours after the best opener"],
            "this word": [int(r.cluster_size), int(r.dup_letters), round(100 * r.min_letter_freq, 1), round(2 ** r.opener_bucket)],
            "average word": [round(diff_df.cluster_size.mean(), 1), round(diff_df.dup_letters.mean(), 2),
                             round(100 * diff_df.min_letter_freq.mean(), 1), round(float((2 ** diff_df.opener_bucket).mean()))],
        })
        st.dataframe(feat, hide_index=True, width="stretch")

        with st.expander("Model details: decision-tree rules and cross-validation"):
            st.code(rep["tree_rules"])
            st.dataframe(pd.DataFrame({"model": list(r2), "CV R²": [round(v, 3) for v in r2.values()]}), hide_index=True)
        st.caption("Difficulty is measured for these solvers, not for human players. Humans use word familiarity and intuition that this analysis doesn't model.")

# ==============================================================================================
# TAB 5 - how it works
# ==============================================================================================
with tab_about:
    st.subheader("How it works")
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        st.markdown(pill("Information theory", "#2f7fd8"), unsafe_allow_html=True)
        st.markdown("#### Entropy solver")
        st.markdown(
            "1. Keep the list of answers that match all feedback so far.\n"
            "2. For every possible guess, work out how the remaining answers would split across the 243 colour patterns.\n"
            "3. Play the guess with the highest **expected information** (entropy, in bits): the one that splits the list most evenly.\n\n"
            "It is a classical decision-making algorithm, an optimisation with no learning involved. "
            "It averages about **3.46 guesses** on all answers."
        )
    with b, st.container(border=True):
        st.markdown(pill("Machine learning", GROUP_COLORS["RL assisted"]), unsafe_allow_html=True)
        st.markdown("#### Reinforcement-learning agent")
        st.markdown(
            "1. A small neural network scores each word in a shortlist of 48 from features (letter statistics, and for "
            "the *assisted* agent also entropy features).\n"
            "2. **PPO** trains it by playing about 38,000 games: −1 per guess, a bonus for information gained.\n"
            "3. The *blind* agent sees only letter statistics, the *assisted* agent also sees entropy features.\n\n"
            "It is evaluated on answers it never trained on."
        )
    st.markdown("#### Where this app gets its numbers")
    st.markdown(
        "- **Solve a game / Watch a word** run the trained solvers live.\n"
        "- **Results** come from `results/eval_all.csv`, every solver playing every answer.\n"
        "- **Why words are hard** comes from `scripts/difficulty.py`, which regresses each word's difficulty on interpretable features."
    )
    st.markdown(
        "#### Limits to keep in mind\n"
        "- The RL agents choose from a shortlist ranked by a prior, so the assisted agent is RL on top of an entropy prior, not RL from scratch.\n"
        "- Word lists are the original 2,309-answer Wordle lists with a uniform prior; newer NYT answers may be missing.\n"
        f"- Source code and write-up: [{REPO}]({REPO})"
    )
