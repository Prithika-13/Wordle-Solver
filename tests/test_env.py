import numpy as np
from gymnasium.utils.env_checker import check_env

from src.env import WordleEnv
from src.features import FeatureBuilder


def test_env_passes_gym_checker():
    env = WordleEnv(FeatureBuilder(mode="blind"), seed=0)
    check_env(env, skip_render_check=True)


def test_env_episode_terminates_and_rewards():
    env = WordleEnv(FeatureBuilder(mode="blind"), seed=1, shaping=0.0)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(0)
        total += r
        done = term or trunc
    assert env.turn <= 6
    assert total <= -1.0


def test_guessing_the_answer_wins():
    fb = FeatureBuilder(mode="blind")
    env = WordleEnv(fb, shaping=0.0)
    env.reset(options={"answer": 5})
    # the answer is always shortlisted once it is a candidate; find it
    env.step(0)  # burn a turn
    cand = env.cand
    j = [i for i, p in enumerate(fb.pool[env._sel]) if p == fb.ans_g[5]]
    if j:
        _, r, term, _, info = env.step(j[0])
        assert term and info["solved"]
