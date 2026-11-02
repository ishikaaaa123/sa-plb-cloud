#!/usr/bin/env python3
"""End-to-end pipeline: training log -> model -> threshold selection (validation seeds)
-> final paired evaluation (test seeds) -> data files, results tables, plots, report.

    python run_all.py --zenodo /path/dataset-replica-mcr-mt.csv          # full run
    python run_all.py --zenodo ... --quick                               # small smoke run
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, time
from dataclasses import asdict
import numpy as np
import pandas as pd

from sa_plb.config import SimConfig
from sa_plb.experiment import (FEATS, build_training_log, make_policies, metrics, obs_frame,
                               routing_frame, run_scenario)
from sa_plb.ml import train_predictor
from sa_plb.policies import MLRouting, RoundRobin, SAPLB, WeightedRoundRobin
from sa_plb.workload import capacity_rps, load_shape

ap = argparse.ArgumentParser()
ap.add_argument("--zenodo", default=None)
ap.add_argument("--out", default=".")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--skip-validation", action="store_true",
                help="reuse results/threshold_selection.json and validation_threshold_sweep.csv from an earlier run")
args = ap.parse_args()
OUT, DATA, RES = args.out, os.path.join(args.out, "data"), os.path.join(args.out, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

cfg = SimConfig()
EVAL_MIN, TRAIN_MIN, PEAK_UTIL = (8, 8, 0.85) if args.quick else (20, 20, 0.85)
TRAIN_DELAY = 0.10   # model is trained under HEALTHY monitoring, then deployed under degraded monitoring
_u, _i = [0.55, 0.75, 0.95, 0.65, 0.85, 0.60, 0.80, 0.80], [0.0, 0.10, 0.20, 0.10, 0.0, 0.20, 0.10, 0.10]
TRAIN_PERIODS = [(_u[k], _i[k], 900 + k) for k in range(8)]
THRESHOLDS = [0.5, 1.0, 2.0, 4.0]
VAL_SEEDS = [100] if args.quick else [100, 101]
TEST_SEEDS = [0] if args.quick else [0, 1, 2]
INTERFS = [0.0, 0.10] if args.quick else [0.0, 0.10, 0.25]   # hidden slowdown episodes / replica / minute
DELAYS = [0.1, 4.0] if args.quick else [0.1, 1.5, 4.0]       # mean extra monitoring delivery delay (s)

curve_eval, curve_train = load_shape(args.zenodo, EVAL_MIN), load_shape(args.zenodo, TRAIN_MIN)
log(f"capacity ~{capacity_rps(cfg):.1f} req/s; WRR weights {cfg.weights()}")

# 1) training log + model --------------------------------------------------------------
tl = build_training_log(cfg, curve_train, TRAIN_PERIODS, TRAIN_DELAY)
tl.to_csv(os.path.join(DATA, "training_log.csv"), index=False)
predictor, mm = train_predictor(tl, FEATS)
json.dump(mm, open(os.path.join(RES, "model_metrics.json"), "w"), indent=2)
log(f"model: test MAE {mm['test_mae_ms']:.0f} ms vs median-baseline {mm['test_baseline_median_mae_ms']:.0f} ms")


def grid(seeds, thresholds, main_thr=None):
    rows = []
    for inter in INTERFS:
        for d in DELAYS:
            for s in seeds:
                tr, _, rs = run_scenario(cfg, curve_eval, PEAK_UTIL, s, d, make_policies(cfg, thresholds),
                                         predictor, interference_per_min=inter)
                for r in rs:
                    m = metrics(r, tr, cfg, main_thr if main_thr else 1.0)
                    m.update(interference_per_min=inter, delay_mean_s=d, seed=s, threshold=r.threshold)
                    rows.append(m)
        log(f"  done interference={inter}")
    return pd.DataFrame(rows)


# 2) choose ONE global SA-PLB threshold on VALIDATION seeds only ------------------------------
if args.skip_validation:
    MAIN_THR = float(json.load(open(os.path.join(RES, "threshold_selection.json")))["chosen_threshold_s"])
    log(f"reusing validation-selected threshold {MAIN_THR}s")
else:
    log("validation sweep for threshold selection")
    val = grid(VAL_SEEDS, THRESHOLDS)
    val.to_csv(os.path.join(RES, "validation_threshold_sweep.csv"), index=False)
    sa = val[val.policy.str.startswith("SA-PLB")]
    per_thr = sa.groupby("threshold").mean_ms.mean()
    MAIN_THR = float(per_thr.idxmin())
    log(f"chosen threshold (lowest mean response across validation conditions): {MAIN_THR}s")
    json.dump(dict(chosen_threshold_s=MAIN_THR, mean_response_ms_by_threshold=per_thr.to_dict(),
                   selected_on="validation seeds %s only" % VAL_SEEDS),
              open(os.path.join(RES, "threshold_selection.json"), "w"), indent=2)

# 3) final evaluation on TEST seeds ----------------------------------------------------------
log("final test evaluation")
test = grid(TEST_SEEDS, THRESHOLDS, MAIN_THR)
test["is_main_sa_plb"] = test.threshold == MAIN_THR
test.to_csv(os.path.join(RES, "test_results_by_seed.csv"), index=False)
test["variant"] = np.where(test.policy.str.startswith("SA-PLB"),
                           np.where(test.is_main_sa_plb, "SA-PLB (main)", test.policy), test.policy)
keys = ["interference_per_min", "delay_mean_s", "variant"]
summ = test.groupby(keys, sort=False).agg(
    mean_ms=("mean_ms", "mean"), mean_ms_sd=("mean_ms", "std"), p95_ms=("p95_ms", "mean"),
    p99_ms=("p99_ms", "mean"), throughput_rps=("throughput_rps", "mean"),
    fallback_rate=("fallback_rate", "mean"), pred_mae_ms=("pred_mae_ms", "mean"),
    age_max_mean_s=("age_max_mean_s", "mean"), fresh_share=("fresh_share", "mean"),
    fresh_mean_ms=("fresh_mean_ms", "mean"), stale_mean_ms=("stale_mean_ms", "mean")).reset_index()
summ.to_csv(os.path.join(RES, "test_summary.csv"), index=False)

# paired differences (same seed/condition => same requests): main SA-PLB minus baseline
pv = test[test.variant.isin(["SA-PLB (main)", "WRR", "ML", "RR"])].pivot_table(
    index=["interference_per_min", "delay_mean_s", "seed"], columns="variant", values="mean_ms")
pairs = []
for (inter, d), g in pv.groupby(level=[0, 1]):
    for b in ["WRR", "ML", "RR"]:
        diff = g["SA-PLB (main)"] - g[b]
        pairs.append(dict(interference_per_min=inter, delay_mean_s=d, baseline=b,
                          mean_diff_ms=diff.mean(), sa_plb_better_in=f"{(diff < 0).sum()}/{len(diff)} seeds"))
pd.DataFrame(pairs).to_csv(os.path.join(RES, "paired_differences.csv"), index=False)

# 4) default scenario with full logs ----------------------------------------------------------
D_INTER, D_DELAY, D_SEED = 0.10, 1.5, 0
pols = [RoundRobin(cfg), WeightedRoundRobin(cfg), MLRouting(), SAPLB(cfg, MAIN_THR)]
tr, sched, rs = run_scenario(cfg, curve_eval, PEAK_UTIL, D_SEED, D_DELAY, pols, predictor,
                             record_obs=True, interference_per_min=D_INTER)
pd.DataFrame(dict(request_id=np.arange(len(tr)), arrival_time=tr.arrival, task_size_kb=tr.size_kb.round(1),
                  request_type=[cfg.type_names[i] for i in tr.type_idx])).to_csv(
    os.path.join(DATA, "requests.csv"), index=False)
pd.concat([routing_frame(r, tr, r.policy, D_SEED, cfg, MAIN_THR) for r in rs]).to_csv(
    os.path.join(DATA, "routing_results.csv"), index=False)
pd.concat([obs_frame(r, r.policy) for r in rs]).to_csv(os.path.join(DATA, "replica_metrics.csv"), index=False)

prov = [
 ("requests.csv","arrival_time","SIMULATED","Poisson arrivals; rate follows Zenodo per-minute record counts (sqrt-compressed)"),
 ("requests.csv","task_size_kb / request_type","SIMULATED","log-normal size, GET/POST/PUT mix (config.py)"),
 ("replica_metrics.csv","collected_at / available_at","SIMULATED","monitor sample time / time it reaches the balancer (periodic + random delay + occasional stalls)"),
 ("replica_metrics.csv","active_requests, queue_length, cpu_pct, mem_pct, recent_avg_response_ms","SIMULATED","true replica state at collected_at, computed by the queue model (+ small noise on cpu/mem)"),
 ("routing_results.csv","selected_replica_id, policy_branch, fallback_used","POLICY OUTPUT","decision made by RR / WRR / ML / SA-PLB"),
 ("routing_results.csv","metric_age_max_s, metric_age_mean_s","DERIVED","decision_time - collected_at of the observation each policy actually saw"),
 ("routing_results.csv","predicted_*","MODEL OUTPUT","regressor prediction (ms); empty when the decision did not use ML"),
 ("routing_results.csv","queue_wait_ms, service_ms, actual_response_ms, completion_time","SIMULATED","exact FCFS multi-server queue; response = wait + service"),
 ("training_log.csv","all feature columns + response_ms","SIMULATED","random-routing logging run; features = decision-time observations only"),
 ("training_log.csv","obs_age_s","DERIVED","logged for analysis; NOT a model feature"),
 ("(hidden, not exported)","service noise, interference slowdown","SIMULATED","never visible to any router or to the model"),
 ("(everything)","no field is a real measurement","-","all values come from simulation driven by a public load SHAPE"),
]
pd.DataFrame(prov, columns=["file", "field", "origin", "how_produced"]).to_csv(
    os.path.join(DATA, "field_provenance.csv"), index=False)
json.dump(dict(config=asdict(cfg), eval_minutes=EVAL_MIN, train_minutes=TRAIN_MIN, peak_util=PEAK_UTIL,
               train=dict(periods=TRAIN_PERIODS, delay=TRAIN_DELAY),
               thresholds=THRESHOLDS, val_seeds=VAL_SEEDS, test_seeds=TEST_SEEDS, interferences=INTERFS,
               delays=DELAYS, main_threshold=MAIN_THR, default_scenario=dict(
                   interference=D_INTER, delay=D_DELAY, seed=D_SEED)),
          open(os.path.join(RES, "run_config.json"), "w"), indent=2, default=str)
log("data files written")

# 5) plots -------------------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = ["RR", "WRR", "ML", "SA-PLB (main)"]
fig, axes = plt.subplots(1, len(INTERFS), figsize=(4.6 * len(INTERFS), 3.8), sharey=False)
for ax, inter in zip(np.atleast_1d(axes), INTERFS):
    for v in order:
        g = summ[(summ.interference_per_min == inter) & (summ.variant == v)]
        ax.errorbar(g.delay_mean_s, g.mean_ms, yerr=g.mean_ms_sd, marker="o", capsize=3, label=v)
    ax.set_yscale("log"); ax.set_xlabel("mean monitoring delay (s)")
    ax.set_title(f"interference {inter}/min/replica"); ax.grid(alpha=.3)
np.atleast_1d(axes)[0].set_ylabel("mean response time (ms, log)"); np.atleast_1d(axes)[0].legend(fontsize=8)
fig.suptitle(f"Test seeds {TEST_SEEDS}: mean response time by policy (SIMULATED)"); fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_policies.png"), dpi=140); plt.close(fig)

fig, axes = plt.subplots(len(INTERFS), len(DELAYS), figsize=(3.6 * len(DELAYS), 2.8 * len(INTERFS)), squeeze=False)
for i, inter in enumerate(INTERFS):
    for j, d in enumerate(DELAYS):
        ax = axes[i][j]
        g = test[(test.interference_per_min == inter) & (test.delay_mean_s == d)]
        s_ = g[g.policy.str.startswith("SA-PLB")].groupby("threshold").agg(m=("mean_ms", "mean"), f=("fallback_rate", "mean"))
        ax.plot(s_.index, s_.m, "o-", label="SA-PLB")
        for b, st in (("WRR", "--"), ("ML", ":")):
            ax.axhline(g[g.policy == b].mean_ms.mean(), ls=st, color="gray" if b == "WRR" else "k", label=b)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"interf {inter}, delay {d}s", fontsize=9)
        if i == len(INTERFS) - 1: ax.set_xlabel("staleness threshold (s)")
        if j == 0: ax.set_ylabel("mean resp. (ms)")
        if i == 0 and j == 0: ax.legend(fontsize=7)
fig.suptitle("SA-PLB threshold sensitivity (test seeds; SIMULATED)"); fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_threshold_sensitivity.png"), dpi=140); plt.close(fig)

# 6) report ------------------------------------------------------------------------------------
def md(df, floatfmt=1):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)

rep = ["# Results report (all values SIMULATED)\n",
       f"Chosen SA-PLB threshold: **{MAIN_THR} s**, selected on validation seeds {VAL_SEEDS} only. "
       f"Final numbers use separate test seeds {TEST_SEEDS}. Model test MAE {mm['test_mae_ms']:.0f} ms "
       f"(median-predictor baseline {mm['test_baseline_median_mae_ms']:.0f} ms).\n",
       "## Mean / P95 response time (ms), mean over test seeds\n"]
t = summ[summ.variant.isin(order)][["interference_per_min", "delay_mean_s", "variant", "mean_ms", "p95_ms", "p99_ms",
                                    "throughput_rps", "fallback_rate", "age_max_mean_s"]]
rep += [md(t, 2), "\n## Paired difference: SA-PLB (main) minus baseline, mean response (ms; negative = SA-PLB better)\n",
        md(pd.DataFrame(pairs), 1)]
open(os.path.join(RES, "REPORT.md"), "w").write("\n".join(rep))
log("done")
