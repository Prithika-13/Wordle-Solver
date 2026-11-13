"""Per-word features describing how useful each guess in the action pool is, given the candidates.

The policy network scores every pool word with the SAME small MLP on these features, so it is
word-agnostic and generalises to answers it never saw in training.

Two feature sets:
  * "blind"    - cheap letter statistics only (coverage / positional frequency of the remaining
                 candidates). No look-ahead; the agent has to learn the strategy.
  * "assisted" - blind + information-theoretic features (entropy, expected remaining size,
                 largest bucket). Lets the agent combine the entropy signal with learned rules.
"""
from __future__ import annotations

import numpy as np

from src.feedback import _to_arr, get_matrix
from src.solvers.entropy import entropy_scores

N_GLOBAL = 3  # log2 n_cand, turn, 1/n_cand
BLIND = ["is_cand", "coverage", "pos_match", "distinct", "dup_letters"]
ASSISTED = BLIND + ["entropy", "exp_remaining", "max_bucket", "uniq_buckets"]


class FeatureBuilder:
    def __init__(self, pool_size_extra: int = 512, mode: str = "assisted", k: int = 48, n_cand_slots: int = 8):
        assert mode in ("blind", "assisted")
        self.mode = mode
        self.k = k  # shortlist size (= action space size seen by the policy)
        self.n_cand_slots = n_cand_slots
        self.M, self.answers, self.guesses = get_matrix()
        na = len(self.answers)
        gi = {w: i for i, w in enumerate(self.guesses)}
        self.ans_g = np.array([gi[w] for w in self.answers])
        # pool = all answers + the non-answer words with the highest opening entropy
        h0 = entropy_scores(self.M, np.arange(na))
        mask = np.ones(len(self.guesses), bool)
        mask[self.ans_g] = False
        extra = np.where(mask)[0]
        extra = extra[np.argsort(-h0[extra])[:pool_size_extra]]
        self.pool = np.concatenate([self.ans_g, extra])  # guess indices; first na are the answers
        self.pool_words = [self.guesses[i] for i in self.pool]
        self.V = len(self.pool)
        self.Mp = np.ascontiguousarray(self.M[self.pool])  # [V, na]
        # letter structure
        A = _to_arr(self.answers) - 97
        P = _to_arr(self.pool_words) - 97
        self.A_pos = np.zeros((na, 5, 26), np.float32)
        for p in range(5):
            self.A_pos[np.arange(na), p, A[:, p]] = 1
        self.A_pres = np.minimum(self.A_pos.sum(1), 1)  # [na,26]
        self.P_pos = np.zeros((self.V, 5, 26), np.float32)
        for p in range(5):
            self.P_pos[np.arange(self.V), p, P[:, p]] = 1
        self.P_pres = np.minimum(self.P_pos.sum(1), 1)
        self.P_distinct = self.P_pres.sum(1) / 5.0
        self.P_dups = 1.0 - self.P_distinct
        self.names = BLIND if mode == "blind" else ASSISTED
        self.n_feat = len(self.names)
        self._h_cache: dict = {}

    @property
    def n_input(self) -> int:
        return self.n_feat + N_GLOBAL

    def build(self, cand: np.ndarray, turn: int, key=None) -> np.ndarray:
        """Return float32 [V, n_feat + N_GLOBAL]."""
        n = len(cand)
        V = self.V
        F = np.zeros((V, self.n_input), np.float32)
        is_cand = np.zeros(V, np.float32)
        is_cand[cand] = 1.0  # pool[:na] are the answers in order
        freq_pres = self.A_pres[cand].mean(0)  # [26]
        freq_pos = self.A_pos[cand].mean(0)  # [5,26]
        coverage = self.P_pres @ freq_pres / 5.0
        pos_match = (self.P_pos * freq_pos[None]).sum((1, 2)) / 5.0
        cols = [is_cand, coverage, pos_match, self.P_distinct, self.P_dups]
        if self.mode == "assisted":
            ck = (key, ) if key is not None else None
            if ck is not None and ck in self._h_cache:
                ent, exp_rem, max_b, uniq = self._h_cache[ck]
            else:
                sub = self.Mp[:, cand].astype(np.int32)
                sub += (np.arange(V, dtype=np.int32) * 243)[:, None]
                counts = np.bincount(sub.ravel(), minlength=V * 243).reshape(V, 243)
                p = counts / n
                with np.errstate(divide="ignore", invalid="ignore"):
                    ent = -np.where(p > 0, p * np.log2(p), 0.0).sum(1)
                exp_rem = (counts * counts).sum(1) / n  # E[|bucket|]
                max_b = counts.max(1)
                uniq = (counts > 0).sum(1)
                ent = ent.astype(np.float32)
                exp_rem = (np.log2(exp_rem) / 12.0).astype(np.float32)
                max_b = (np.log2(max_b) / 12.0).astype(np.float32)
                uniq = (uniq / 243.0).astype(np.float32)
                if ck is not None and turn <= 1 and len(self._h_cache) < 2000:
                    self._h_cache[ck] = (ent, exp_rem, max_b, uniq)
            cols += [ent / 8.0, exp_rem, max_b, uniq]
        for j, c in enumerate(cols):
            F[:, j] = c
        F[:, self.n_feat + 0] = np.log2(n) / 12.0
        F[:, self.n_feat + 1] = turn / 6.0
        F[:, self.n_feat + 2] = 1.0 / n
        if self.mode == "assisted":
            self._prior = cols[5] + is_cand / n  # bits + candidate tie-break (same as EntropySolver)
        else:
            self._prior = coverage + pos_match + 0.2 * self.P_distinct + 0.1 * is_cand
        self._is_cand = is_cand
        return F

    def state(self, cand: np.ndarray, turn: int, key=None) -> tuple[np.ndarray, np.ndarray]:
        """Shortlist the K most promising pool words. Returns (obs [K, n_input], pool_idx [K])."""
        F = self.build(cand, turn, key)
        order = np.argsort(-self._prior)
        cand_order = order[self._is_cand[order] > 0][: self.n_cand_slots]
        head = order[: self.k - len(cand_order)]
        sel = list(dict.fromkeys(list(head) + list(cand_order)))  # unique, keeps order
        if len(sel) < self.k:
            seen = set(sel)
            sel += [i for i in order[self.k + 8 : self.k + 8 + 4 * self.k] if i not in seen][: self.k - len(sel)]
        sel = np.array(sel[: self.k])
        return F[sel], sel
