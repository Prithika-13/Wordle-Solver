"""Word-difficulty features: what makes a Wordle answer hard to solve?

Every feature is cheap to compute from the answer list alone (no solver needed), so a model
trained on them tells us *why* a word is hard rather than just *that* it is.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.feedback import get_matrix, load_words

VOWELS = set("aeiou")

FEATURE_LABELS = {
    "cluster_size": "Look-alike cluster size (words sharing 4 letters in place)",
    "n_neighbors": "Total one-letter neighbours",
    "dup_letters": "Repeated letters",
    "mean_letter_freq": "Letter commonness (mean)",
    "min_letter_freq": "Rarest letter's frequency",
    "pos_score": "Positional letter-frequency score",
    "n_vowels": "Number of vowels",
    "opener_bucket": "Words sharing its colours after the best opener (log2)",
}


def _clusters(words: list[str]) -> dict[tuple[int, str], list[str]]:
    """Group words that agree on 4 positions and differ only at position p, e.g. (0, 'atch')."""
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for w in words:
        for p in range(5):
            groups[(p, w[:p] + w[p + 1 :])].append(w)
    return groups


def word_features(answers: list[str] | None = None) -> pd.DataFrame:
    """One row per answer with interpretable difficulty drivers."""
    if answers is None:
        answers, _ = load_words()
    n = len(answers)
    groups = _clusters(answers)
    letter_presence = defaultdict(int)
    pos_counts = [defaultdict(int) for _ in range(5)]
    for w in answers:
        for ch in set(w):
            letter_presence[ch] += 1
        for p, ch in enumerate(w):
            pos_counts[p][ch] += 1

    # how crowded is the word's bucket after the best opener?  (soare = highest-entropy opener)
    M, ans, guesses = get_matrix()
    from src.solvers.entropy import entropy_scores

    h0 = entropy_scores(M, np.arange(len(ans)))
    opener = int(np.argmax(h0))
    codes = M[opener]
    bucket_size = np.bincount(codes, minlength=243)[codes]
    bucket = dict(zip(ans, bucket_size))

    rows = []
    for w in answers:
        cl = []
        for p in range(5):
            members = [x for x in groups[(p, w[:p] + w[p + 1 :])] if x != w]
            cl.append((len(members), p, members))
        best = max(cl, key=lambda t: t[0])
        n_neighbors = sum(c[0] for c in cl)
        freqs = [letter_presence[ch] / n for ch in set(w)]
        pat = "".join(w[i] if i != best[1] else "_" for i in range(5))
        rows.append(
            dict(
                word=w,
                cluster_size=best[0] + 1 if best[0] else 0,  # including the word itself when a cluster exists
                cluster_pattern=pat if best[0] else "",
                cluster_members=", ".join(best[2][:12]),
                n_neighbors=n_neighbors,
                dup_letters=5 - len(set(w)),
                mean_letter_freq=float(np.mean(freqs)),
                min_letter_freq=float(np.min(freqs)),
                pos_score=float(sum(pos_counts[p][ch] for p, ch in enumerate(w)) / (5 * n)),
                n_vowels=sum(c in VOWELS for c in w),
                opener_bucket=float(np.log2(bucket[w])),
            )
        )
    return pd.DataFrame(rows)
