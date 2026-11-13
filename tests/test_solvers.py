import numpy as np

from src.feedback import ALL_GREEN, get_matrix
from src.solvers.entropy import EntropySolver, entropy_scores
from src.solvers.heuristic import FrequencyHeuristic
from src.solvers.random_solver import RandomSolver


def test_matrix_shape_and_diagonal():
    M, answers, guesses = get_matrix()
    assert M.shape == (len(guesses), len(answers))
    gi = {w: i for i, w in enumerate(guesses)}
    for j in range(0, len(answers), 97):
        assert M[gi[answers[j]], j] == ALL_GREEN


def test_entropy_bounds():
    M, answers, _ = get_matrix()
    h = entropy_scores(M, np.arange(len(answers)))
    assert (h >= 0).all() and h.max() <= np.log2(243) + 1e-6
    assert 5.5 < h.max() < 6.5  # best opener carries ~5.9 bits


def test_entropy_two_candidates_zero_for_nonseparating():
    M, answers, _ = get_matrix()
    h = entropy_scores(M, np.array([0, 1]))
    assert h.max() <= 1.0 + 1e-9


def test_solvers_win_sample():
    s_ent = EntropySolver()
    for solver in (s_ent, FrequencyHeuristic(), RandomSolver(0)):
        wins = 0
        for j in range(0, 2309, 100):
            g = solver.play(j, max_turns=10)
            wins += solver.solved(g, j)
        assert wins >= 20  # of 24 (random may miss a few)
    for j in range(0, 2309, 50):
        g = s_ent.play(j, max_turns=6)
        assert s_ent.solved(g, j), s_ent.answers[j]


def test_candidate_filtering_consistent():
    s = EntropySolver()
    g = s.play(42)
    assert s.solved(g, 42)
