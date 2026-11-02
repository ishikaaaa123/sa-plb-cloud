# SA-PLB: Staleness-Aware Predictive Load Balancing under Delayed Monitoring

SA-PLB is a simulated load-balancing system that uses a machine-learning (ML) response-time predictor when monitoring information is sufficiently fresh. If the monitoring information is older than a configured freshness threshold, SA-PLB falls back to Weighted Round Robin (WRR).

The project compares four routing policies: Round Robin (RR), Weighted Round Robin (WRR), ML-based Predictive Load Balancing (ML), and Staleness-Aware Predictive Load Balancing (SA-PLB).

> **Data and simulation statement**
>
> This project uses a **hybrid simulation approach**. The public Zenodo trace (`dataset-replica-mcr-mt.csv`) is used only to derive a *relative workload-intensity pattern*. The current implementation counts records in each one-minute `timestamp_2` bucket, square-root compresses those counts, and selects the most bursty 20-minute window. These record counts are a workload proxy; they are **not treated as measured incoming request counts**.
>
> Individual request arrivals are generated synthetically using a non-homogeneous Poisson process guided by that relative pattern. Request sizes, request types, replica resources, queues, monitoring observations and delays, interference episodes, and response times are simulated. No performance result in this project is a measurement from real containers or from the Alibaba Cloud cluster.

## 1. Policies compared

All policies are evaluated using the same generated request trace and the same monitoring schedule within each paired scenario.

| Policy | Behaviour |
|---|---|
| **RR** | Sends requests to replicas in rotation. |
| **WRR** | Uses smooth weighted round robin with static replica weights based on configured cores and speed. |
| **ML** | Predicts the response time for each replica using the latest monitoring observation already delivered to the balancer, then selects the replica with the lowest predicted response time. It does not apply a freshness check. |
| **SA-PLB** | Uses the same ML prediction when the oldest observation used for the decision is within the configured freshness threshold; otherwise, it falls back to WRR. |

## 2. How the simulation works

### Request and queue simulation

- Requests arrive according to a non-homogeneous Poisson process. Its changing arrival rate follows the relative workload-intensity pattern derived from the Zenodo trace.
- Request sizes and request types are generated synthetically.
- Each replica has configured processing capacity.
- Each replica is modelled as an FCFS (First-Come, First-Served) queue.
- A request's response time is calculated as queue waiting time plus service time.

### Monitoring and staleness

- Each replica is monitored approximately every 0.5 seconds, with a small sampling-time jitter.
- A monitoring sample describes the replica state at `collected_at`.
- The sample reaches the load balancer at `available_at`, after a simulated delivery delay. Occasional additional monitoring stalls are also simulated.
- The age of an observation at routing decision time is:

  `observation_age = decision_time - collected_at`

- The SA-PLB decision uses the maximum age among the observations used for that routing decision. If this age is greater than the selected threshold, SA-PLB uses WRR.

### Interference

The simulation can introduce hidden slowdown episodes for individual replicas. Each episode lasts approximately 20–60 seconds and multiplies service time by 2.0–3.5. The configured experiment tests interference rates of 0, 0.10, and 0.25 episodes per replica per minute. These are simulated conditions, not measurements from a production system.

### ML model

- Model: `HistGradientBoostingRegressor`.
- Target: log-transformed response time; predictions are converted back to milliseconds.
- Training examples are collected using random routing so that the training log covers different replicas and system states.
- The model uses decision-time monitoring features, request information, and static replica configuration.
- Observation age is deliberately excluded from the base ML feature set so that the plain ML policy does not itself account for freshness.
- The chronological split is performed by independent simulation period: the first six periods are training, the seventh is validation, and the eighth is test.

## 3. Dataset interpretation and provenance

The Zenodo file is **not used as a request-level trace**. In `sa_plb/workload.py`, the implementation reads column index 1 as the time bucket, counts records per bucket, applies square-root compression, and normalizes the selected window to a peak of 1.0.

| Component | Source |
|---|---|
| Relative workload-intensity pattern | Derived from record counts per one-minute bucket in the Zenodo trace; used as a relative workload proxy only. |
| Individual request arrivals | Simulated using a non-homogeneous Poisson process guided by the relative pattern. |
| Request sizes and request types | Simulated. |
| Replica resources and queues | Simulated. |
| Monitoring collection and delivery delays | Simulated. |
| Interference episodes | Simulated. |
| Response times and performance metrics | Produced by the simulator. |

**Important limitation:** Because the workload pattern is derived from record counts rather than verified request-rate measurements, it should not be described as real request traffic or as the actual Alibaba cluster arrival rate.

## 4. Experimental configuration

The full experiment included:

| Setting | Value |
|---|---|
| Number of replicas | 3 |
| Evaluation workload window | 20 minutes |
| Training workload window | 20 minutes per period |
| Nominal monitoring interval | 0.5 seconds |
| Training periods | 8 independent periods |
| Training / validation / test split | First 6 / seventh / eighth period |
| Validation seeds | 100, 101 |
| Test seeds | 0, 1, 2 |
| Interference rates | 0, 0.10, 0.25 episodes per replica per minute |
| Mean monitoring delays tested | 0.1, 1.5, 4.0 seconds |
| Candidate SA-PLB thresholds | 0.5, 1.0, 2.0, 4.0 seconds |
| Selected threshold in the included full run | **4.0 seconds** |

The threshold was selected by choosing the candidate with the lowest mean SA-PLB response time across the validation conditions. In the included full run, the validation means were approximately 4,650 ms (0.5 s), 3,861 ms (1.0 s), 2,227 ms (2.0 s), and 1,487 ms (4.0 s). The selected value is therefore 4.0 seconds for this run; it is not a universal or production-ready threshold.

## 5. Results from the included full run

The following are **simulated** mean response times in milliseconds, averaged over test seeds 0, 1, and 2. SA-PLB results use the selected 4.0-second threshold.

| Interference rate | Mean monitoring delay (s) | RR | WRR | ML | SA-PLB |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.1 | 2,234 | 333 | 480 | 480 |
| 0.00 | 1.5 | 2,234 | 333 | 771 | 765 |
| 0.00 | 4.0 | 2,234 | 333 | 1,107 | 1,010 |
| 0.10 | 0.1 | 7,481 | 5,679 | 672 | 672 |
| 0.10 | 1.5 | 7,481 | 5,679 | 1,073 | 1,065 |
| 0.10 | 4.0 | 7,481 | 5,679 | 1,429 | 1,301 |
| 0.25 | 0.1 | 24,052 | 14,282 | 1,823 | 1,823 |
| 0.25 | 1.5 | 24,052 | 14,282 | 2,436 | 2,302 |
| 0.25 | 4.0 | 24,052 | 14,282 | 2,738 | 2,708 |

### How to interpret these results

- In this run, increasing monitoring delay generally increases the mean response time of the ML-based policies.
- With no interference, WRR has a lower mean response time than ML and SA-PLB in all three tested delay conditions.
- Under interference, ML and SA-PLB have lower mean response times than RR and WRR in the tested scenarios.
- SA-PLB is close to ML in many conditions and is somewhat better in several delayed-monitoring conditions. The improvement is not universal or large in every scenario.
- At the selected 4.0-second threshold, the mean fallback rate is approximately 8% in the 4.0-second-delay scenarios in this run, and 0% in the other tested delay scenarios. The fallback mechanism therefore activates only in a subset of decisions under these settings.

These findings describe this simulator and its chosen assumptions. They do not establish that SA-PLB will outperform other load balancers in a real cloud deployment.

## 6. Model prediction metrics and an important evaluation caveat

The included full run reports:

| Metric | Result |
|---|---:|
| Training samples | 54,749 |
| Validation samples | 9,962 |
| Test samples | 8,700 |
| Validation MAE for selected model | approximately 6,356 ms |
| Test MAE | approximately 120 ms |
| Test RMSE | approximately 247 ms |
| Test median-predictor baseline MAE | approximately 264 ms |

**The validation/test MAE gap is large and must be treated as a limitation, not hidden.** The validation period uses peak utilization 0.80 with interference rate 0.10, while the test period uses peak utilization 0.70 with no interference. These periods therefore represent different operating conditions. The low test MAE does not demonstrate equivalent prediction accuracy under interference or under all delayed-monitoring scenarios.

The next evaluation improvement should make validation and test operating conditions more comparable, then regenerate the model and all dependent results before updating these figures.

## 7. Files and project structure

```text
run_all.py                 Full experiment pipeline
app.py                     Streamlit demonstration dashboard
sa_plb/
    config.py              Simulation configuration
    workload.py            Workload generation and monitoring schedule
    sim.py                 Queue and event simulation
    policies.py            RR, WRR, ML and SA-PLB policies
    ml.py                  ML feature construction and model training
    experiment.py           Training log, scenarios and metrics
tests/
    test_sa_plb.py          Unit tests
data/                       Generated data files from a run
results_full/
    data/                    Data from the included full run
    results/
        REPORT.md            Detailed generated report
        test_summary.csv
        test_results_by_seed.csv
        paired_differences.csv
        validation_threshold_sweep.csv
        threshold_selection.json
        model_metrics.json
        run_config.json
        fig_policies.png
        fig_threshold_sensitivity.png
results_quick/               Outputs from the quick/smoke run
```

Generated files may be updated when the pipeline is run again. The values in this README refer specifically to the included `results_full` run.

## 8. Reproduce the experiment

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib streamlit
```

Run the full pipeline, passing the path to the Zenodo CSV:

```bash
python run_all.py --zenodo /path/to/dataset-replica-mcr-mt.csv
```

Run a shorter smoke test:

```bash
python run_all.py --zenodo /path/to/dataset-replica-mcr-mt.csv --quick
```

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

To demonstrate the dashboard:

```bash
streamlit run app.py
```

### Important reproduction notes

- **Always pass `--zenodo` when running `run_all.py`.** If the path is omitted or the file cannot be used, `load_shape` falls back to a built-in synthetic workload shape. That produces a different experiment and should not be compared directly with the included Zenodo-derived results.
- The current `app.py` calls `load_shape(None, ...)`, so the dashboard uses the built-in synthetic workload shape rather than the Zenodo-derived shape. It is a demonstration interface and does **not** reproduce the exact full-run results in this README.
- `--skip-validation` reuses the threshold stored in the existing results directory. Use it only when intentionally reusing a previously selected threshold; otherwise run validation again.
- A full run may take substantially longer than a quick run, depending on the machine.

## 9. Limitations and future improvements

1. **Workload proxy:** Zenodo record counts are used as a relative workload-intensity proxy, not verified incoming request counts.
2. **Simulation assumptions:** Request arrivals, service times, replica resources, monitoring delays, interference and response times are simulated.
3. **Validation/test mismatch:** The included validation and test periods have different utilization and interference conditions, contributing to the unusually large MAE gap. A comparable split and rerun are needed.
4. **Limited seeds:** The included final evaluation uses three test seeds; no statistical significance test is reported.
5. **Threshold scope:** One global threshold is selected for the simulated validation conditions. It should not be treated as an optimal threshold for real deployments.
6. **Fallback choice:** WRR is the only fallback policy evaluated.
7. **Feature comparison:** The base ML model does not use observation age as a feature. An age-aware ML baseline could be explored in future work.
8. **Distribution shift:** The model is trained using random-routing logs but deployed under policy-driven routing, which can change the states encountered by the model.
9. **Dashboard workload:** The current dashboard uses a synthetic fallback workload and is not an exact reproduction of the Zenodo-derived full experiment.

## 10. Conclusion

SA-PLB adds an explicit freshness check to predictive load balancing: it uses ML predictions when the monitoring observations are within a configured age threshold and falls back to WRR otherwise. The included simulation demonstrates how monitoring delay and interference can affect routing-policy performance.

The results are scenario-dependent. SA-PLB does not outperform WRR or plain ML in every condition. The main contribution demonstrated by this prototype is the implementation and evaluation of a staleness-aware routing decision, not a claim of universal performance superiority.
# sa-plb-cloud
