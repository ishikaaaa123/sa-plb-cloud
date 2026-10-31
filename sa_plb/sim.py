"""Discrete-event simulation of N replicas behind a load balancer.

Queue model (exact FCFS multi-server): a replica with c cores keeps the time each core
becomes free. A request arriving at t starts at max(t, earliest free core), runs for its
(replica-specific) service time, and finishes after that. So:
    response = queue_wait + service,   and every routing choice changes later queue waits.

Monitoring model: each replica's monitor samples its TRUE state at `collected_at`
(only requests that arrived by then, completions that finished by then) and the
observation reaches the balancer at `available_at >= collected_at`. At a decision time t
the router sees, per replica, the latest observation (by collected_at) with available_at <= t.
Staleness = t - collected_at, computed from timestamps.
"""
import heapq
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import SimConfig
from .ml import BASE_FEATURES, candidate_features
from .workload import MonitorSchedule, Trace


class Obs:
    __slots__ = ("obs_id", "replica", "collected_at", "available_at", "active", "queue",
                 "cpu", "mem", "recent_rt_ms")

    def __init__(self, obs_id, replica, collected_at, available_at, active, queue, cpu, mem, rt):
        self.obs_id, self.replica = obs_id, replica
        self.collected_at, self.available_at = collected_at, available_at
        self.active, self.queue, self.cpu, self.mem, self.recent_rt_ms = active, queue, cpu, mem, rt


class ReplicaState:
    def __init__(self, spec, window: int):
        self.spec = spec
        self.free = [0.0] * spec.cores          # time each core becomes free
        self.pending = []                       # heap of (finish, start, response_s) of unfinished requests
        self.recent = deque(maxlen=window)      # response times (s) of the latest completed requests

    def advance(self, t: float):
        while self.pending and self.pending[0][0] <= t:
            _, _, resp = heapq.heappop(self.pending)
            self.recent.append(resp)

    def state(self, t: float):
        """True state at time t (only used by the monitor and for tests, never by routers)."""
        self.advance(t)
        active = len(self.pending)
        queue = sum(1 for _, s, _ in self.pending if s > t)
        recent = 1000.0 * float(np.mean(self.recent)) if self.recent else 0.0
        return active, queue, active - queue, recent

    def assign(self, t: float, service: float):
        j = min(range(len(self.free)), key=self.free.__getitem__)
        start = max(t, self.free[j])
        finish = start + service
        self.free[j] = finish
        heapq.heappush(self.pending, (finish, start, finish - t))
        return start, finish


def build_events(trace: Trace, sched: MonitorSchedule):
    """(time, kind, a, b): kind 0 = monitor collection (a=replica, b=k), kind 1 = arrival (a=request)."""
    ev = []
    for r, times in enumerate(sched.collect):
        ev.extend((float(t), 0, r, k) for k, t in enumerate(times))
    ev.extend((float(t), 1, i, 0) for i, t in enumerate(trace.arrival))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


@dataclass
class RunResult:
    policy: str
    threshold: Optional[float]
    n: int
    replica: np.ndarray
    wait_s: np.ndarray
    service_s: np.ndarray
    resp_s: np.ndarray
    completion: np.ndarray
    age_max: np.ndarray
    age_mean: np.ndarray
    fallback: np.ndarray
    branch: List[str]
    preds_ms: np.ndarray            # (n, R) predictions for each candidate; NaN when not an ML decision
    feat_chosen: np.ndarray         # (n, F) observation features of the chosen replica
    chosen_obs_collected: np.ndarray
    obs_log: list                   # monitor observations (only when record_obs=True)


def run_policy(policy, trace: Trace, events, sched: MonitorSchedule, cfg: SimConfig,
               predictor=None, record_obs: bool = False) -> RunResult:
    R, n = cfg.n_replicas, len(trace)
    states = [ReplicaState(sp, cfg.recent_window) for sp in cfg.replicas]
    incoming = [[] for _ in range(R)]          # heaps of (available_at, k, Obs): delivered, not yet visible
    cur: List[Optional[Obs]] = [None] * R
    out = dict(rep=np.zeros(n, int), wait=np.zeros(n), svc=np.zeros(n), resp=np.zeros(n),
               comp=np.zeros(n), amax=np.zeros(n), amean=np.zeros(n), fb=np.zeros(n, bool),
               preds=np.full((n, R), np.nan), feat=np.zeros((n, len(BASE_FEATURES))),
               oc=np.zeros(n))
    branch, obs_log = [""] * n, []
    for t, kind, a, b in events:
        if kind == 0:
            st = states[a]
            spec = cfg.replicas[a]
            active, queue, busy, recent = st.state(t)
            cpu = float(np.clip(100.0 * busy / spec.cores + sched.cpu_noise[a][b], 0, 100))
            mem = float(np.clip(cfg.mem_base_pct + 100.0 * active * cfg.mem_per_active_mb / spec.mem_mb
                                + sched.mem_noise[a][b], 0, 100))
            o = Obs(b, a, t, float(sched.avail[a][b]), active, queue, cpu, mem, recent)
            heapq.heappush(incoming[a], (o.available_at, b, o))
            if record_obs:
                obs_log.append(o)
            continue
        i = a
        for r in range(R):                     # reveal observations that have arrived by now
            h = incoming[r]
            while h and h[0][0] <= t:
                o = heapq.heappop(h)[2]
                if cur[r] is None or o.collected_at > cur[r].collected_at:
                    cur[r] = o
        ages = [t - o.collected_at for o in cur]
        size, tidx = float(trace.size_kb[i]), int(trace.type_idx[i])
        d = policy.choose(t, size, tidx, cur, ages, cfg, predictor)
        r = d.replica
        service = trace.service_s(cfg, i, r)
        start, finish = states[r].assign(t, service)
        out["rep"][i], out["wait"][i], out["svc"][i] = r, start - t, service
        out["resp"][i], out["comp"][i] = finish - t, finish
        out["amax"][i], out["amean"][i] = max(ages), float(np.mean(ages))
        out["fb"][i], branch[i] = d.fallback, d.branch
        if d.preds is not None:
            out["preds"][i] = d.preds
        out["feat"][i] = candidate_features(cfg, size, tidx, cur)[r]
        out["oc"][i] = cur[r].collected_at
    return RunResult(policy.name, getattr(policy, "threshold", None), n, out["rep"], out["wait"],
                     out["svc"], out["resp"], out["comp"], out["amax"], out["amean"], out["fb"],
                     branch, out["preds"], out["feat"], out["oc"], obs_log)
