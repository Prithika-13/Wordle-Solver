import random

import numpy as np

from src.feedback import ALL_GREEN, compute_matrix, decode, encode, feedback, load_words


def fb(g, a):
    return decode(feedback(g, a))


def test_all_green():
    assert feedback("crane", "crane") == ALL_GREEN


def test_all_gray():
    assert fb("fuzzy", "crane") == "-----"


def test_duplicate_guess_letter_one_in_answer():
    # answer ABIDE has one E: first E yellow, second E gray
    assert fb("speed", "abide") == "--Y-Y"


def test_green_consumes_letter_before_yellow():
    # the green E at the end uses up ABIDE's only E, so earlier Es are gray
    assert fb("eerie", "abide") == "---YG"


def test_green_and_yellow_same_letter():
    # LEVER has two Es: pos1 green, pos0 yellow (second E), pos4 gray
    assert fb("eerie", "lever") == "YGY--"


def test_repeated_s():
    assert fb("sassy", "glass") == "YY-G-"


def test_repeated_l_and_e():
    assert fb("allee", "apple") == "GY--G"


def test_encode_roundtrip():
    for c in range(243):
        assert encode(decode(c)) == c


def test_matrix_matches_reference():
    answers, guesses = load_words()
    rng = random.Random(0)
    gs = rng.sample(guesses, 40)
    As = rng.sample(answers, 40)
    m = compute_matrix(gs, As)
    for i, g in enumerate(gs):
        for j, a in enumerate(As):
            assert m[i, j] == feedback(g, a), (g, a)


def test_matrix_tricky_words():
    ws = ["speed", "abide", "eerie", "lever", "sassy", "glass", "allee", "apple", "geese", "elder", "llama", "mamma"]
    m = compute_matrix(ws, ws)
    for i, g in enumerate(ws):
        for j, a in enumerate(ws):
            assert m[i, j] == feedback(g, a), (g, a)
