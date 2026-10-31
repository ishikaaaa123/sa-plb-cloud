"""The four routing policies (+ a random policy used ONLY to collect ML training logs).

A policy sees only decision-time information: the request's size/type, the latest
monitoring observation per replica that has ALREADY ARRIVED, static replica config,
and (for ML) a trained predictor. It never sees true current replica state.
"""
from collections import namedtuple
from typing import List, Optional, Sequence

import numpy as np

from .config import SimConfig
from .ml import candidate_features

Decision = namedtuple("Decision", "replica branch preds fallback")


class SmoothWRR:
    """Smooth weighted round robin (nginx style). Deterministic given the weights."""

    def __init__(self, weights: Sequence[int]):
        self.w = list(weights)
        self.cur = [0] * len(self.w)
        self.total = sum(self.w)

    def next(self) -> int:
        for i, w in enumerate(self.w):
            self.cur[i] += w
        j = max(range(len(self.w)), key=lambda i: (self.cur[i], -i))
        self.cur[j] -= self.total
        return j


class Policy:
    name = "base"
    threshold: Optional[float] = None

    def choose(self, t, size_kb, type_idx, obs, ages, cfg: SimConfig, predictor) -> Decision:
        raise NotImplementedError


class RoundRobin(Policy):
    name = "RR"

    def __init__(self, cfg: SimConfig):
        self.n, self.k = cfg.n_replicas, 0

    def choose(self, t, size_kb, type_idx, obs, ages, cfg, predictor):
        r = self.k % self.n
        self.k += 1
        return Decision(r, "rr", None, False)


class WeightedRoundRobin(Policy):
    name = "WRR"

    def __init__(self, cfg: SimConfig):
        self.wrr = SmoothWRR(cfg.weights())

    def choose(self, t, size_kb, type_idx, obs, ages, cfg, predictor):
        return Decision(self.wrr.next(), "wrr", None, False)


class MLRouting(Policy):
    """Predict response time on every candidate from the (possibly stale) observations;
    pick the lowest prediction. Ignores freshness."""
    name = "ML"

    def choose(self, t, size_kb, type_idx, obs, ages, cfg, predictor):
        X = candidate_features(cfg, size_kb, type_idx, obs, ages)
        preds = predictor.predict(X)
        return Decision(int(np.argmin(preds)), "ml", preds, False)


class SAPLB(Policy):
    """Staleness-aware: ML if the observations used are fresh, otherwise WRR fallback.

    Decision-level age = max age over the candidates' observations (conservative:
    one stale candidate makes the whole comparison unreliable). Fresh iff age <= threshold.
    """

    def __init__(self, cfg: SimConfig, threshold_s: float):
        self.threshold = float(threshold_s)
        self.name = f"SA-PLB@{threshold_s:g}s"
        self.wrr = SmoothWRR(cfg.weights())

    def choose(self, t, size_kb, type_idx, obs, ages, cfg, predictor):
        if max(ages) <= self.threshold:
            X = candidate_features(cfg, size_kb, type_idx, obs, ages)
            preds = predictor.predict(X)
            return Decision(int(np.argmin(preds)), "sa_ml", preds, False)
        return Decision(self.wrr.next(), "sa_fallback_wrr", None, True)


class RandomLogging(Policy):
    """Uniform random routing -- used only to generate unbiased ML training logs."""
    name = "RANDOM(log)"

    def __init__(self, cfg: SimConfig, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n = cfg.n_replicas

    def choose(self, t, size_kb, type_idx, obs, ages, cfg, predictor):
        return Decision(int(self.rng.integers(self.n)), "random", None, False)
