# SA-PLB: Staleness-Aware Predictive Load Balancer (simulation)

SA-PLB is a load balancer that uses a machine-learning response-time predictor **only when its monitoring
data is fresh**, and falls back to Weighted Round Robin (WRR) when the data is stale.

> **Data statement.** The dataset is a **hybrid**. The only real, public input is the *shape* of the request
> load over time, taken from a Zenodo trace (`dataset-replica-mcr-mt.csv`). Replicas, queues, monitoring
> delays, interference and response times are all **simulated**. No value is a measurement from real containers.

## Policies compared (same requests and monitoring schedule for each)

| Policy | Behaviour |
|---|---|
| RR | cycles through replicas |
| WRR | smooth weighted round robin, static weights (4, 4, 2) from cores x speed |
| ML | predicts response time per replica from the latest delivered observation, picks the lowest; ignores freshness |
| SA-PLB | same ML if the **oldest** observation used is at most the threshold old, otherwise WRR |

## How it works

- **Queue model:** exact FCFS multi-server queue per replica; response = queue wait + service time.
- **Monitoring:** each replica is sampled about every 0.5 s; the sample is the true state at `collected_at` and
  reaches the balancer at `available_at` after a random delay (plus occasional 5 s stalls).
  **Staleness = decision time - collected_at.**
- **Interference:** hidden slowdown episodes (2 to 3.5x slower for 20 to 60 s) per replica, at 0, 0.10 and 0.25 per minute.
- **Model:** `HistGradientBoostingRegressor` on log response time. Features are decision-time only; observation age
  is deliberately not a feature. Trained on random-routing logs from 8 periods under healthy monitoring,
  chronological split by period (last = test, one before = validation).
- **Threshold:** one global value chosen on validation seeds (100, 101) only: **2.0 s**. Final results use test seeds 0, 1, 2.

## Hybrid dataset: what is real and what is simulated

| Part | Origin |
|---|---|
| Load shape over time | REAL: records per 1-minute bucket from the Zenodo file (column index 1), square-root compressed, most bursty 20-minute window. The file's columns are undocumented, so no claim is made about what a record means. |
| Everything else (arrivals, sizes, queues, monitoring, interference, response times) | SIMULATED |

## Files

```
run_all.py            full pipeline (training log -> model -> threshold -> test evaluation -> data, results, plots)
app.py                dashboard (see warning below)
sa_plb/               config.py, workload.py, sim.py, policies.py, ml.py, experiment.py
tests/test_sa_plb.py  17 unit tests (queue logic, staleness, both SA-PLB branches, leakage, splits)
data/                 training_log.csv (47,051 rows), requests.csv (6,854), replica_metrics.csv (30,276),
                      routing_results.csv (27,416 = 4 policies x 6,854), field_provenance.csv
results/              REPORT.md, test_summary.csv, paired_differences.csv, test_results_by_seed.csv,
                      validation_threshold_sweep.csv, threshold_selection.json, model_metrics.json,
                      run_config.json, fig_policies.png, fig_threshold_sensitivity.png
```

## Reproduce

```bash
pip install numpy pandas scikit-learn matplotlib
python run_all.py --zenodo /path/to/dataset-replica-mcr-mt.csv   # about 35 min on 1 CPU
python run_all.py --zenodo /path/to/dataset-replica-mcr-mt.csv --quick   # about 3 min smoke run
python -m unittest discover -s tests -v
```

`--skip-validation` reuses the saved threshold from `results/threshold_selection.json`.

> **Warning: always pass `--zenodo`.** If no path is given, `load_shape` silently falls back to a built-in
> synthetic load shape and produces different numbers (for example WRR 5,679 ms instead of 1,014 ms at 0.10
> interference). `app.py` currently calls `load_shape(None, ...)`, so the dashboard also runs on the synthetic
> shape and will not reproduce the numbers below.

## Main results (mean response time in ms, mean of 3 test seeds, threshold 2.0 s)

| Interference | Delay (s) | RR | WRR | ML | SA-PLB |
|---|---|---|---|---|---|
| 0 | 0.1 | 1,077 | 302 | 397 | 396 |
| 0 | 1.5 | 1,077 | 302 | 597 | 495 |
| 0 | 4.0 | 1,077 | 302 | 748 | 359 |
| 0.10 | 0.1 | 1,306 | 1,014 | 454 | 458 |
| 0.10 | 1.5 | 1,306 | 1,014 | 669 | 586 |
| 0.10 | 4.0 | 1,306 | 1,014 | 899 | 553 |
| 0.25 | 0.1 | 5,940 | 4,830 | 627 | 635 |
| 0.25 | 1.5 | 5,940 | 4,830 | 1,004 | 929 |
| 0.25 | 4.0 | 5,940 | 4,830 | 1,595 | 2,731 |

- Staleness hurts ML: with no interference its mean rises from 397 ms (0.1 s delay) to 748 ms (4 s delay).
- SA-PLB is faster than ML in all 3 seeds in 5 of 6 stale conditions (75 to 389 ms) and matches ML when data is fresh.
- It is worse than ML with heavy interference and 4 s delay (80% of decisions fall back to WRR, which handles interference badly).
- WRR wins in every seed without interference. With interference SA-PLB beats WRR on average, but at 0.10 only in 2 of 3 seeds.
- Model test MAE 175 ms vs 236 ms for a median predictor (RMSE 599 vs 610 ms). The test period has no interference, so this is optimistic for interference scenarios.

## Limitations

- All performance values are simulated; interference rates, delays, service times and replica specs are assumptions, not calibrated to a real system.
- Only 3 test seeds and no significance tests.
- One global threshold; 2.0 s and 4.0 s were nearly tied on validation (627 vs 632 ms).
- Only WRR was tested as a fallback, and there is no baseline where ML receives observation age as a feature.
- The model is trained under random routing but deployed under its own routing (distribution shift).
