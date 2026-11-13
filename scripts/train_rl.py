"""PPO training of a word-scoring policy on the Wordle environment.

    python scripts/train_rl.py --mode assisted --seed 0 --iters 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.env import WordleEnv
from src.features import FeatureBuilder
from src.solvers.rl_agent import WordPolicy, save_policy


def split_answers(n: int, test_frac: float = 0.2, seed: int = 123):
    """Fixed train/test split of the answer list (same for every run)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(n * test_frac)
    return np.sort(perm[k:]), np.sort(perm[:k])  # train, test


@torch.no_grad()
def rollout(env: WordleEnv, net: WordPolicy, n_episodes: int, rng: np.random.Generator):
    obs_l, act_l, logp_l, ret_l = [], [], [], []
    ep_scores, ep_solved = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_obs, ep_act, ep_lp, ep_rew = [], [], [], []
        done = False
        while not done:
            logits, _ = net(torch.from_numpy(obs)[None])
            dist = torch.distributions.Categorical(logits=logits[0])
            a = dist.sample()
            nobs, r, term, trunc, info = env.step(int(a))
            ep_obs.append(obs); ep_act.append(int(a)); ep_lp.append(float(dist.log_prob(a))); ep_rew.append(r)
            obs, done = nobs, term or trunc
        G = np.cumsum(ep_rew[::-1])[::-1]  # undiscounted reward-to-go (episodes are <= 6 steps)
        obs_l += ep_obs; act_l += ep_act; logp_l += ep_lp; ret_l += list(G)
        ep_solved.append(info["solved"]); ep_scores.append(len(ep_rew) if info["solved"] else 7)
    return (
        torch.from_numpy(np.stack(obs_l)),
        torch.tensor(act_l),
        torch.tensor(logp_l, dtype=torch.float32),
        torch.tensor(ret_l, dtype=torch.float32),
        float(np.mean(ep_scores)),
        float(np.mean(ep_solved)),
    )


@torch.no_grad()
def greedy_eval(env: WordleEnv, net: WordPolicy, answers: np.ndarray) -> tuple[float, float]:
    scores, solved = [], []
    for a in answers:
        obs, _ = env.reset(options={"answer": int(a)})
        done = False
        n = 0
        while not done:
            logits, _ = net(torch.from_numpy(obs)[None])
            obs, r, term, trunc, info = env.step(int(logits.argmax(-1)))
            n += 1
            done = term or trunc
        solved.append(info["solved"]); scores.append(n if info["solved"] else 7)
    return float(np.mean(scores)), float(np.mean(solved))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="assisted", choices=["blind", "assisted"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=128, help="episodes per PPO iteration")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--k", type=int, default=48, help="shortlist size / action space")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shaping", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fb = FeatureBuilder(mode=args.mode, k=args.k)
    train_idx, test_idx = split_answers(len(fb.answers))
    env = WordleEnv(fb, answer_pool=train_idx, shaping=args.shaping, seed=args.seed)
    eval_env = WordleEnv(fb)
    net = WordPolicy(fb.n_input, args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    tag = args.out or f"rl_{args.mode}_s{args.seed}"
    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    log = []
    t0 = time.time()
    best = (9.0, None)
    for it in range(1, args.iters + 1):
        obs, act, old_lp, ret, sc, sr = rollout(env, net, args.episodes, rng)
        with torch.no_grad():
            _, v = net(obs)
        adv = ret - v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(act)
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, args.minibatch):
                mb = perm[s : s + args.minibatch]
                logits, value = net(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act[mb])
                ratio = (lp - old_lp[mb]).exp()
                pg = -torch.min(ratio * adv[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[mb]).mean()
                vl = F.mse_loss(value, ret[mb])
                loss = pg + 0.5 * vl - args.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        row = dict(iter=it, episodes=it * args.episodes, train_mean_guesses=sc, train_solve_rate=sr, seconds=time.time() - t0)
        if it % args.eval_every == 0 or it == args.iters:
            tm, ts = greedy_eval(eval_env, net, test_idx)
            row.update(test_mean_guesses=tm, test_solve_rate=ts)
            print(f"[{args.mode} s{args.seed}] it {it:4d} train(sampled) {sc:.3f} | test(greedy) {tm:.3f} solve {ts:.3f} | {time.time()-t0:.0f}s", flush=True)
            if tm < best[0]:
                best = (tm, {k: v.clone() for k, v in net.state_dict().items()})
        log.append(row)
    pd.DataFrame(log).to_csv(out_dir / f"{tag}_log.csv", index=False)
    save_policy(net, out_dir / f"{tag}.pt", args.mode, fb.n_input, args.hidden, fb.k, dict(args=vars(args)))
    print("saved", out_dir / f"{tag}.pt")


if __name__ == "__main__":
    main()
