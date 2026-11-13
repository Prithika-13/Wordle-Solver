"""Interactive helper: enter your guesses + the colours Wordle gave you, get the best next guess.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from src.feedback import decode, encode
from src.solvers.entropy import EntropySolver

st.set_page_config(page_title="Wordle Solver", page_icon="🟩")
COLORS = {"G": "#538d4e", "Y": "#b59f3b", "-": "#3a3a3c"}


@st.cache_resource
def solver():
    return EntropySolver()


s = solver()
st.title("Wordle solver")
st.caption("Greedy information-theoretic solver: picks the guess with the highest expected information.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (word, pattern)

cand = np.arange(len(s.answers))
for w, pat in st.session_state.history:
    code = encode(pat)
    cand = cand[s.M[s.guess_index[w], cand] == code]

for w, pat in st.session_state.history:
    st.markdown(
        "".join(
            f"<span style='display:inline-block;width:2.2em;height:2.2em;line-height:2.2em;text-align:center;"
            f"margin:2px;color:white;font-weight:700;background:{COLORS[p]}'>{c.upper()}</span>"
            for c, p in zip(w, pat)
        ),
        unsafe_allow_html=True,
    )

st.write(f"**{len(cand)}** possible answers remain.")
if len(cand) == 0:
    st.error("No word matches that feedback - check the colours you entered.")
elif len(cand) == 1:
    st.success(f"The answer is **{s.answers[cand[0]].upper()}**")
else:
    top = s.top_k(cand, 5)
    st.subheader("Suggested guesses")
    st.table(
        [{"word": w.upper(), "expected bits": round(h, 3), "could be the answer": c} for w, h, c in top]
    )
    if len(cand) <= 12:
        st.write("Remaining: " + ", ".join(s.answers[i] for i in cand))

with st.form("add", clear_on_submit=True):
    w = st.text_input("Word you guessed", max_chars=5).lower()
    pat = st.text_input("Feedback (G = green, Y = yellow, - = gray), e.g. GY--G", max_chars=5).upper()
    if st.form_submit_button("Add"):
        if len(w) == 5 and w in s.guess_index and len(pat) == 5 and set(pat) <= set("GY-"):
            st.session_state.history.append((w, pat))
            st.rerun()
        else:
            st.warning("Enter a valid 5-letter word and a 5-character G/Y/- pattern.")

if st.button("New game"):
    st.session_state.history = []
    st.rerun()
