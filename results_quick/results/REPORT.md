# Results report (all values SIMULATED)

Chosen SA-PLB threshold: **0.5 s**, selected on validation seeds [100] only. Final numbers use separate test seeds [0]. Model test MAE 156 ms (median-predictor baseline 438 ms).

## Mean / P95 response time (ms), mean over test seeds

| interference_per_min | delay_mean_s | variant | mean_ms | p95_ms | p99_ms | throughput_rps | fallback_rate | age_max_mean_s |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.10 | RR | 2320.83 | 13089.51 | 15148.57 | 11.49 | 0.00 | 0.59 |
| 0.00 | 0.10 | WRR | 346.40 | 834.46 | 1299.67 | 11.49 | 0.00 | 0.59 |
| 0.00 | 0.10 | ML | 535.40 | 1393.47 | 2141.00 | 11.49 | 0.00 | 0.59 |
| 0.00 | 0.10 | SA-PLB (main) | 383.30 | 925.27 | 1400.27 | 11.49 | 0.70 | 0.59 |
| 0.00 | 4.00 | RR | 2320.83 | 13089.51 | 15148.57 | 11.49 | 0.00 | 2.73 |
| 0.00 | 4.00 | WRR | 346.40 | 834.46 | 1299.67 | 11.49 | 0.00 | 2.73 |
| 0.00 | 4.00 | ML | 1249.97 | 3860.03 | 6352.04 | 11.47 | 0.00 | 2.73 |
| 0.00 | 4.00 | SA-PLB (main) | 346.40 | 834.46 | 1299.67 | 11.49 | 1.00 | 2.73 |
| 0.10 | 0.10 | RR | 4569.38 | 15385.60 | 58808.69 | 10.34 | 0.00 | 0.59 |
| 0.10 | 0.10 | WRR | 1215.62 | 7991.92 | 20317.88 | 11.49 | 0.00 | 0.59 |
| 0.10 | 0.10 | ML | 554.42 | 1419.37 | 2325.03 | 11.49 | 0.00 | 0.59 |
| 0.10 | 0.10 | SA-PLB (main) | 558.11 | 1208.21 | 7865.87 | 11.49 | 0.70 | 0.59 |
| 0.10 | 4.00 | RR | 4569.38 | 15385.60 | 58808.69 | 10.34 | 0.00 | 2.73 |
| 0.10 | 4.00 | WRR | 1215.62 | 7991.92 | 20317.88 | 11.49 | 0.00 | 2.73 |
| 0.10 | 4.00 | ML | 1392.51 | 4295.48 | 8454.04 | 11.45 | 0.00 | 2.73 |
| 0.10 | 4.00 | SA-PLB (main) | 1215.62 | 7991.92 | 20317.88 | 11.49 | 1.00 | 2.73 |

## Paired difference: SA-PLB (main) minus baseline, mean response (ms; negative = SA-PLB better)

| interference_per_min | delay_mean_s | baseline | mean_diff_ms | sa_plb_better_in |
|---|---|---|---|---|
| 0.0 | 0.1 | WRR | 36.9 | 0/1 seeds |
| 0.0 | 0.1 | ML | -152.1 | 1/1 seeds |
| 0.0 | 0.1 | RR | -1937.5 | 1/1 seeds |
| 0.0 | 4.0 | WRR | 0.0 | 0/1 seeds |
| 0.0 | 4.0 | ML | -903.6 | 1/1 seeds |
| 0.0 | 4.0 | RR | -1974.4 | 1/1 seeds |
| 0.1 | 0.1 | WRR | -657.5 | 1/1 seeds |
| 0.1 | 0.1 | ML | 3.7 | 0/1 seeds |
| 0.1 | 0.1 | RR | -4011.3 | 1/1 seeds |
| 0.1 | 4.0 | WRR | 0.0 | 0/1 seeds |
| 0.1 | 4.0 | ML | -176.9 | 1/1 seeds |
| 0.1 | 4.0 | RR | -3353.8 | 1/1 seeds |