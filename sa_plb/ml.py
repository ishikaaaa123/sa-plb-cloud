"""Decision-time features + response-time predictor.

Every feature below is available to the router at decision time:
  * obs_*            -> latest monitoring observation that has already arrived
  * task_size_kb, type_cost -> known from the request
  * replica_cores, replica_speed, est_service_ms -> static configuration + request
NOT used: true active requests, the request's own response time, future information.
Observation age is excluded by default so plain ML really "ignores freshness".
"""
from typing import List, Sequence

import numpy as np

BASE_FEATURES = ["obs_active", "obs_queue", "obs_cpu_pct", "obs_mem_pct", "obs_recent_rt_ms",
                 "task_size_kb", "type_cost", "replica_cores", "replica_speed", "est_service_ms"]
FEATURES_WITH_AGE = BASE_FEATURES + ["obs_age_s"]
FORBIDDEN_FEATURES = {"true_active_requests", "actual_response_ms", "response_time_ms",
                      "queue_wait_ms", "service_ms", "completion_time"}


def candidate_features(cfg, size_kb, type_idx, obs, ages=None, include_age=False) -> np.ndarray:
    rows = []
    cost = cfg.type_cost[type_idx]
    work = cfg.base_service_s + cfg.per_kb_service_s * size_kb
    for r, rep in enumerate(cfg.replicas):
        o = obs[r]
        row = [o.active, o.queue, o.cpu, o.mem, o.recent_rt_ms,
               size_kb, cost, rep.cores, rep.speed, work * cost / rep.speed * 1000.0]
        if include_age:
            row.append(ages[r])
        rows.append(row)
    return np.asarray(rows, dtype=float)


class Predictor:
    """Wraps a fitted regressor trained on log(response_ms); returns milliseconds."""

    def __init__(self, model, feature_names: Sequence[str]):
        self.model = model
        self.feature_names = list(feature_names)
        assert not (set(self.feature_names) & FORBIDDEN_FEATURES), "leaky feature"

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.model.predict(X))


def train_predictor(df, feature_names: Sequence[str], seed: int = 0):
    """df has feature columns, 'response_ms' and a chronological 'split' column
    ('train' < 'val' < 'test'). Hyper-parameters are chosen on VAL only; TEST is
    reported once at the end. Returns (Predictor, metrics dict)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, ytr = tr[list(feature_names)].to_numpy(), np.log(tr.response_ms.to_numpy())
    grid = [dict(max_iter=100, learning_rate=0.10, max_depth=4),
            dict(max_iter=200, learning_rate=0.05, max_depth=6),
            dict(max_iter=150, learning_rate=0.10, max_depth=None, max_leaf_nodes=31),
            dict(max_iter=300, learning_rate=0.05, max_depth=4)]
    best, best_mae, tried = None, np.inf, []
    for hp in grid:
        m = HistGradientBoostingRegressor(random_state=seed, **hp).fit(Xtr, ytr)
        pv = np.exp(m.predict(va[list(feature_names)].to_numpy()))
        mae = float(np.mean(np.abs(pv - va.response_ms.to_numpy())))
        tried.append({**{k: (v if v is not None else "None") for k, v in hp.items()}, "val_mae_ms": mae})
        if mae < best_mae:
            best, best_mae, best_hp = m, mae, hp
    pt = np.exp(best.predict(te[list(feature_names)].to_numpy()))
    yt = te.response_ms.to_numpy()
    med = float(np.median(tr.response_ms))
    metrics = dict(
        n_train=len(tr), n_val=len(va), n_test=len(te),
        chosen_hyperparameters={k: (v if v is not None else "None") for k, v in best_hp.items()},
        validation_grid=tried,
        test_mae_ms=float(np.mean(np.abs(pt - yt))),
        test_rmse_ms=float(np.sqrt(np.mean((pt - yt) ** 2))),
        test_baseline_median_mae_ms=float(np.mean(np.abs(med - yt))),
        test_baseline_median_rmse_ms=float(np.sqrt(np.mean((med - yt) ** 2))),
        features=list(feature_names),
    )
    return Predictor(best, feature_names), metrics
