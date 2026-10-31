"""Training-log generation, model training, paired policy evaluation and summaries."""
import numpy as np
import pandas as pd

from .config import SimConfig
from .ml import BASE_FEATURES, train_predictor
from .policies import (MLRouting, RandomLogging, RoundRobin, SAPLB, WeightedRoundRobin)
from .sim import build_events, run_policy
from .workload import make_monitor_schedule, make_trace

FEATS = BASE_FEATURES


def make_policies(cfg, thresholds):
    return [RoundRobin(cfg), WeightedRoundRobin(cfg), MLRouting()] + [SAPLB(cfg, th) for th in thresholds]


def build_training_log(cfg: SimConfig, curve, periods, delay_mean_s: float):
    """Random routing => unbiased coverage of replicas/states. One row per request: the
    chosen replica's decision-time features and its actual response time.

    `periods` = list of (peak_util, interference_per_min, seed). Each period is an independent
    logging run placed one after another in time. Chronological split BY PERIOD: the last
    period is TEST, the one before is VALIDATION, all earlier ones are TRAIN. (Splitting a
    single run by time would leave the one load burst entirely in val/test.)"""
    frames, offset = [], 0.0
    for k, (util, inter, seed) in enumerate(periods):
        trace = make_trace(cfg, curve, util, seed, interference_per_min=inter)
        sched = make_monitor_schedule(cfg, trace.horizon_s, seed + 10_000, delay_mean_s)
        res = run_policy(RandomLogging(cfg, seed + 20_000), trace, build_events(trace, sched), sched, cfg)
        df = pd.DataFrame(res.feat_chosen, columns=FEATS)
        df.insert(0, "period", k)
        df.insert(1, "decision_time", trace.arrival + offset)
        df.insert(2, "replica_id", res.replica)
        df["period_peak_util"], df["period_interference"] = util, inter
        df["obs_age_s"] = res.age_max      # logged for analysis only; NOT a model feature
        df["response_ms"] = res.resp_s * 1000
        df["split"] = "train" if k < len(periods) - 2 else ("val" if k == len(periods) - 2 else "test")
        frames.append(df)
        offset += trace.horizon_s + 600.0  # gap so periods do not overlap
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "request_id", np.arange(len(out)))
    return out


def run_scenario(cfg, curve, peak_util, seed, delay_mean_s, policies, predictor, record_obs=False,
                 interference_per_min=0.0):
    """One paired comparison: identical requests + identical monitoring schedule for every policy."""
    trace = make_trace(cfg, curve, peak_util, seed, interference_per_min=interference_per_min)
    sched = make_monitor_schedule(cfg, trace.horizon_s, seed + 10_000, delay_mean_s)
    events = build_events(trace, sched)
    return trace, sched, [run_policy(p, trace, events, sched, cfg, predictor, record_obs) for p in policies]


def metrics(res, trace, cfg, fresh_threshold_s):
    rt = res.resp_s * 1000
    makespan = res.completion.max() - trace.arrival.min()
    share = np.bincount(res.replica, minlength=cfg.n_replicas) / res.n
    cap = np.array([r.cores * r.speed for r in cfg.replicas]); cap = cap / cap.sum()
    ml = ~np.isnan(res.preds_ms).all(axis=1)
    pred_chosen = res.preds_ms[np.arange(res.n), res.replica]
    fresh = res.age_max <= fresh_threshold_s
    out = dict(policy=res.policy, n=res.n, mean_ms=rt.mean(), median_ms=np.median(rt),
               p95_ms=np.percentile(rt, 95), p99_ms=np.percentile(rt, 99),
               mean_wait_ms=res.wait_s.mean() * 1000, throughput_rps=res.n / makespan,
               imbalance=float(np.abs(share - cap).sum() / 2), fallback_rate=float(res.fallback.mean()),
               ml_decision_rate=float(ml.mean()),
               pred_mae_ms=float(np.mean(np.abs(pred_chosen[ml] - rt[ml]))) if ml.any() else np.nan,
               pred_rmse_ms=float(np.sqrt(np.mean((pred_chosen[ml] - rt[ml]) ** 2))) if ml.any() else np.nan,
               age_mean_s=res.age_mean.mean(), age_max_mean_s=res.age_max.mean(),
               fresh_share=float(fresh.mean()),
               fresh_mean_ms=rt[fresh].mean() if fresh.any() else np.nan,
               stale_mean_ms=rt[~fresh].mean() if (~fresh).any() else np.nan,
               fresh_p95_ms=np.percentile(rt[fresh], 95) if fresh.any() else np.nan,
               stale_p95_ms=np.percentile(rt[~fresh], 95) if (~fresh).any() else np.nan)
    return out


def routing_frame(res, trace, policy_label, seed, cfg, fresh_threshold_s):
    n = res.n
    df = pd.DataFrame(dict(policy=policy_label, seed=seed, request_id=np.arange(n),
                           decision_time=trace.arrival, task_size_kb=trace.size_kb.round(1),
                           request_type=[cfg.type_names[i] for i in trace.type_idx],
                           selected_replica_id=res.replica, policy_branch=res.branch,
                           fallback_used=res.fallback,
                           staleness_threshold_s=res.threshold if res.threshold is not None else np.nan,
                           metric_age_max_s=res.age_max, metric_age_mean_s=res.age_mean,
                           fresh_under_main_threshold=res.age_max <= fresh_threshold_s,
                           chosen_obs_collected_at=res.chosen_obs_collected))
    for r in range(cfg.n_replicas):
        df[f"predicted_ms_replica{r}"] = res.preds_ms[:, r]
    df["predicted_response_ms"] = res.preds_ms[np.arange(n), res.replica]
    for j, f in enumerate(FEATS[:5]):
        df[f] = res.feat_chosen[:, j]
    df["queue_wait_ms"] = res.wait_s * 1000
    df["service_ms"] = res.service_s * 1000
    df["completion_time"] = res.completion
    df["actual_response_ms"] = res.resp_s * 1000
    return df


def obs_frame(res, policy_label):
    return pd.DataFrame([dict(policy=policy_label, replica_id=o.replica, obs_id=o.obs_id,
                              collected_at=o.collected_at, available_at=o.available_at,
                              delay_s=o.available_at - o.collected_at, active_requests=o.active,
                              queue_length=o.queue, cpu_pct=o.cpu, mem_pct=o.mem,
                              recent_avg_response_ms=o.recent_rt_ms) for o in res.obs_log])
