"""Request-arrival traces and monitoring schedules (all seeded, reusable across policies)."""
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import SimConfig


def load_shape(zenodo_path: Optional[str], minutes: int) -> np.ndarray:
    """Relative load per minute (peak = 1.0).

    ASSUMPTION: the Zenodo file's columns are not documented here, so we only use the
    NUMBER OF RECORDS per 1-minute bucket (column index 1, values are multiples of
    60 000) as a *relative load shape*. We make no claim about what a record means.
    The counts are square-root compressed (otherwise one 1000-record minute dwarfs everything else),
    then the most bursty `minutes`-long window (highest coefficient of variation) is used.
    If the file is missing, a synthetic bursty shape is used instead.
    """
    if zenodo_path and os.path.exists(zenodo_path):
        df = pd.read_csv(zenodo_path, header=None, usecols=[1], names=["t"])
        per_min = df.groupby("t").size()
        full = pd.RangeIndex(0, int(per_min.index.max()) + 60000, 60000)
        s = np.sqrt(per_min.reindex(full).interpolate().bfill().ffill())  # sqrt: keep quiet minutes non-negligible
        if len(s) >= minutes:
            cv = s.rolling(minutes).std() / s.rolling(minutes).mean()
            end = cv.idxmax()
            win = s.loc[end - (minutes - 1) * 60000: end].to_numpy(dtype=float)
            return win / win.max()
    x = np.arange(minutes, dtype=float)
    shape = 0.35 + 0.65 * np.exp(-((x - 0.3 * minutes) / (0.06 * minutes + 1)) ** 2) \
        + 0.8 * np.exp(-((x - 0.72 * minutes) / (0.05 * minutes + 1)) ** 2)
    return shape / shape.max()


def mean_service_s(cfg: SimConfig) -> float:
    mean_size = math.exp(cfg.size_log_mean + cfg.size_log_sigma ** 2 / 2)
    mean_cost = sum(p * c for p, c in zip(cfg.type_prob, cfg.type_cost))
    return (cfg.base_service_s + cfg.per_kb_service_s * mean_size) * mean_cost


def capacity_rps(cfg: SimConfig) -> float:
    return sum(r.cores * r.speed for r in cfg.replicas) / mean_service_s(cfg)


@dataclass
class Trace:
    arrival: np.ndarray      # (n,) seconds, sorted
    size_kb: np.ndarray      # (n,)
    type_idx: np.ndarray     # (n,) int
    noise: np.ndarray        # (n, R) hidden multiplicative service noise (SIMULATED, not visible to routers)
    horizon_s: float
    slow: Optional[np.ndarray] = None   # (n, R) hidden interference slowdown (>=1) at each request's arrival

    def __len__(self):
        return len(self.arrival)

    def service_s(self, cfg: SimConfig, i: int, r: int) -> float:
        rep = cfg.replicas[r]
        work = cfg.base_service_s + cfg.per_kb_service_s * self.size_kb[i]
        m = 1.0 if self.slow is None else self.slow[i, r]
        return float(work * cfg.type_cost[self.type_idx[i]] / rep.speed * self.noise[i, r] * m)


def make_trace(cfg: SimConfig, curve: np.ndarray, peak_util: float, seed: int,
               floor_frac: float = 0.25, interference_per_min: float = 0.0) -> Trace:
    """Non-homogeneous Poisson arrivals following `curve`. peak_util = peak arrival rate / capacity.

    interference_per_min > 0 adds hidden slowdown episodes per replica (SIMULATED 'noisy
    neighbour'): each lasts 20-60 s and multiplies service time by 2.0-3.5. Static WRR
    weights cannot see them; they only show up in (delayed) monitoring data. They are drawn
    from a separate RNG so arrivals/sizes/noise are identical with and without interference."""
    rng = np.random.default_rng(seed)
    peak_rps = peak_util * capacity_rps(cfg)
    chunks: List[np.ndarray] = []
    for m, rel in enumerate(curve):
        nxt = curve[m + 1] if m + 1 < len(curve) else rel
        for sec in range(60):
            frac = (rel + (nxt - rel) * sec / 60.0)
            rate = max(frac, floor_frac) * peak_rps
            k = rng.poisson(rate)
            if k:
                chunks.append(m * 60 + sec + rng.random(k))
    arrival = np.sort(np.concatenate(chunks)) if chunks else np.array([])
    n = len(arrival)
    size = np.clip(rng.lognormal(cfg.size_log_mean, cfg.size_log_sigma, n),
                   cfg.size_min_kb, cfg.size_max_kb)
    tidx = rng.choice(len(cfg.type_names), size=n, p=cfg.type_prob)
    s = cfg.service_noise_sigma
    noise = rng.lognormal(-s * s / 2, s, (n, cfg.n_replicas))
    slow = np.ones((n, cfg.n_replicas))
    if interference_per_min > 0:
        irng = np.random.default_rng(seed + 30_000)
        horizon = len(curve) * 60.0
        for r in range(cfg.n_replicas):
            for _ in range(irng.poisson(interference_per_min * horizon / 60.0)):
                st, du, fa = irng.uniform(0, horizon), irng.uniform(20, 60), irng.uniform(2.0, 3.5)
                slow[(arrival >= st) & (arrival < st + du), r] *= fa
    return Trace(arrival, size, tidx, noise, len(curve) * 60.0, slow)


@dataclass
class MonitorSchedule:
    collect: List[np.ndarray]   # per replica: collection times (sorted)
    avail: List[np.ndarray]     # per replica: time the observation reaches the balancer
    cpu_noise: List[np.ndarray]
    mem_noise: List[np.ndarray]


def make_monitor_schedule(cfg: SimConfig, horizon_s: float, seed: int,
                          delay_mean_s: Optional[float] = None) -> MonitorSchedule:
    """Each replica is sampled periodically; each observation is delivered after a delay.
    Independent of routing, so it is identical for every policy in a paired comparison."""
    rng = np.random.default_rng(seed)
    dmean = cfg.monitor_delay_mean_s if delay_mean_s is None else delay_mean_s
    collect, avail, cn, mn = [], [], [], []
    for _ in cfg.replicas:
        t, times = 0.0, [0.0]                     # first observation at t=0 (empty system)
        while t < horizon_s + 60:
            t += cfg.monitor_interval_s * (1 + rng.uniform(-cfg.monitor_jitter_frac,
                                                           cfg.monitor_jitter_frac))
            times.append(t)
        times = np.array(times)
        k = len(times)
        delay = cfg.monitor_base_delay_s + (rng.exponential(dmean, k) if dmean > 0 else 0.0)
        delay = delay + (rng.random(k) < cfg.stall_prob) * cfg.stall_extra_s
        delay[0] = 0.0                            # warm start: initial observation available at once
        collect.append(times)
        avail.append(times + delay)
        cn.append(rng.normal(0, cfg.cpu_noise_sd, k))
        mn.append(rng.normal(0, cfg.mem_noise_sd, k))
    return MonitorSchedule(collect, avail, cn, mn)
