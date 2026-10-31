
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sa_plb.config import SimConfig
from sa_plb.workload import load_shape, make_trace, make_monitor_schedule
from sa_plb.sim import build_events, run_policy
from sa_plb.policies import (
    RoundRobin,
    WeightedRoundRobin,
    MLRouting,
    SAPLB,
)
from sa_plb.ml import BASE_FEATURES, train_predictor


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="SA-PLB Dashboard",
    page_icon="⚖️",
    layout="wide",
)

ROOT = Path(__file__).parent
DATA = ROOT / "results_full" / "data"

st.title("Staleness-Aware Predictive Load Balancer")
st.caption("Interactive hybrid simulation and performance monitoring")


# --------------------------------------------------
# LOAD ML MODEL
# --------------------------------------------------

@st.cache_resource
def get_predictor():
    df = pd.read_csv(DATA / "training_log.csv")
    return train_predictor(df, BASE_FEATURES)


try:
    predictor, model_metrics = get_predictor()
except Exception as e:
    st.error(f"ML model could not be loaded: {e}")
    st.stop()


# --------------------------------------------------
# SIMULATION SETTINGS
# --------------------------------------------------

cfg = SimConfig()

st.sidebar.header("Simulation Settings")

seed = st.sidebar.number_input(
    "Random seed", min_value=0, value=42, step=1
)

peak_util = st.sidebar.slider(
    "Peak workload utilisation",
    0.30, 0.95, 0.85, 0.05
)

interference = st.sidebar.select_slider(
    "Interference per minute",
    options=[0.0, 0.1, 0.25],
    value=0.1,
)

monitor_delay = st.sidebar.select_slider(
    "Monitoring delay (seconds)",
    options=[0.1, 1.5, 4.0],
    value=1.5,
)

threshold = st.sidebar.select_slider(
    "SA-PLB freshness threshold (seconds)",
    options=[0.5, 1.0, 2.0, 4.0],
    value=4.0,
)


# --------------------------------------------------
# RUN SIMULATION
# --------------------------------------------------

def run_experiment(seed, peak_util, interference,
                   monitor_delay, threshold):

    cfg = SimConfig()

    # Generate hybrid synthetic workload
    curve = load_shape(None, minutes=20)

    trace = make_trace(
        cfg,
        curve,
        peak_util=peak_util,
        seed=int(seed),
        interference_per_min=interference,
    )

    # Generate monitoring observations
    schedule = make_monitor_schedule(
        cfg,
        trace.horizon_s,
        seed=int(seed) + 1000,
        delay_mean_s=monitor_delay,
    )

    events = build_events(trace, schedule)

    # Four policies receive identical workload and monitoring schedule
    policies = [
        RoundRobin(cfg),
        WeightedRoundRobin(cfg),
        MLRouting(),
        SAPLB(cfg, threshold),
    ]

    results = {}

    for policy in policies:
        results[policy.name] = run_policy(
            policy,
            trace,
            events,
            schedule,
            cfg,
            predictor=predictor,
            record_obs=True,
        )

    return trace, results


if st.button("Run Simulation", type="primary"):

    with st.spinner("Running all four routing policies..."):

        try:
            trace, results = run_experiment(
                seed,
                peak_util,
                interference,
                monitor_delay,
                threshold,
            )

            st.session_state["trace"] = trace
            st.session_state["results"] = results
            st.session_state["settings"] = {
                "seed": seed,
                "threshold": threshold,
                "delay": monitor_delay,
                "interference": interference,
            }

        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "results" not in st.session_state:
    st.info("Choose your settings and click Run Simulation.")
    st.stop()

trace = st.session_state["trace"]
results = st.session_state["results"]
settings = st.session_state["settings"]

st.success("Simulation completed successfully.")

st.divider()


# --------------------------------------------------
# SYSTEM OVERVIEW
# --------------------------------------------------

st.subheader("System Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Replicas", cfg.n_replicas)
c2.metric("Routing Policies", 4)
c3.metric("ML Model", "HistGradientBoosting")
c4.metric("Freshness Threshold", f"{settings['threshold']} s")

st.caption(
    f"Requests generated: {len(trace):,} | "
    f"Workload duration: {trace.horizon_s / 60:.0f} minutes"
)


# --------------------------------------------------
# REPLICA CONFIGURATION
# --------------------------------------------------

st.subheader("Replica Configuration")

replica_df = pd.DataFrame([
    {
        "Replica": f"R{r.id}",
        "CPU Cores": r.cores,
        "Relative Speed": r.speed,
        "Memory (MB)": r.mem_mb,
        "WRR Weight": cfg.weights()[r.id],
    }
    for r in cfg.replicas
])

st.dataframe(replica_df, use_container_width=True, hide_index=True)


# --------------------------------------------------
# PERFORMANCE COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("Policy Performance Comparison")

rows = []

for name, result in results.items():

    response_ms = result.resp_s * 1000

    rows.append({
        "Policy": name,
        "Mean Response (ms)": round(np.mean(response_ms), 2),
        "P95 Response (ms)": round(np.percentile(response_ms, 95), 2),
        "P99 Response (ms)": round(np.percentile(response_ms, 99), 2),
        "Mean Queue Wait (ms)": round(np.mean(result.wait_s) * 1000, 2),
        "Fallback (%)": round(np.mean(result.fallback) * 100, 2),
    })

comparison = pd.DataFrame(rows)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Mean Response Time")

chart_df = comparison.set_index("Policy")[["Mean Response (ms)"]]
st.bar_chart(chart_df)


# --------------------------------------------------
# INDIVIDUAL REQUEST INSPECTION
# --------------------------------------------------

st.divider()
st.subheader("Individual Request Routing")

request_index = st.slider(
    "Select request",
    min_value=0,
    max_value=len(trace) - 1,
    value=0,
)

request_type = cfg.type_names[int(trace.type_idx[request_index])]
request_size = trace.size_kb[request_index]
request_time = trace.arrival[request_index]

a, b, c = st.columns(3)

a.metric("Request ID", request_index)
b.metric("Request Type", request_type)
c.metric("Request Size", f"{request_size:.2f} KB")

st.caption(f"Request arrival time: {request_time:.3f} seconds")


# --------------------------------------------------
# REPLICA MONITORING AT REQUEST ARRIVAL
# --------------------------------------------------

st.divider()
st.subheader("Replica Monitoring")

monitoring_policy = st.selectbox(
    "View monitoring data for",
    options=list(results.keys()),
    index=list(results.keys()).index(
        next(k for k in results if k.startswith("SA-PLB@"))
    ),
)

monitor_result = results[monitoring_policy]
observations = monitor_result.obs_log

monitor_cols = st.columns(cfg.n_replicas)

monitor_rows = []

for r in range(cfg.n_replicas):

    # Only observations already delivered when this request arrived
    available = [
        o for o in observations
        if o.replica == r and o.available_at <= request_time
    ]

    if not available:
        with monitor_cols[r]:
            st.warning(f"R{r}: No observation available")
        continue

    # Latest observation by collection time
    obs = max(available, key=lambda o: o.collected_at)

    age = request_time - obs.collected_at
    status = "FRESH" if age <= settings["threshold"] else "STALE"

    monitor_rows.append({
        "Replica": f"R{r}",
        "Active Requests": obs.active,
        "Queue Length": obs.queue,
        "CPU (%)": round(obs.cpu, 2),
        "Memory (%)": round(obs.mem, 2),
        "Recent Avg Response (ms)": round(obs.recent_rt_ms, 2),
        "Metric Age (s)": round(age, 3),
        "Status": status,
    })

    with monitor_cols[r]:
        with st.container(border=True):
            st.subheader(f"Replica R{r}")

            st.metric("CPU Usage", f"{obs.cpu:.1f}%")
            st.metric("Memory Usage", f"{obs.mem:.1f}%")
            st.metric("Active Requests", obs.active)
            st.metric("Queue Length", obs.queue)
            st.metric("Metric Age", f"{age:.3f}s")

            if status == "FRESH":
                st.success("FRESH")
            else:
                st.error("STALE")

if monitor_rows:
    st.subheader("Monitoring Data Table")
    st.dataframe(
        pd.DataFrame(monitor_rows),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# ROUTING DECISIONS
# --------------------------------------------------

decision_rows = []

for name, result in results.items():

    replica = int(result.replica[request_index])
    predictions = result.preds_ms[request_index]

    prediction_text = (
        ", ".join(
            f"R{i}: {p:.2f} ms"
            for i, p in enumerate(predictions)
        )
        if np.all(np.isfinite(predictions))
        else "Not used (WRR fallback or non-ML policy)"
    )

    decision_rows.append({
        "Policy": name,
        "Selected Replica": f"R{replica}",
        "Decision Branch": result.branch[request_index],
        "Metric Age Max (s)": round(
            result.age_max[request_index], 3
        ),
        "Fallback Used": bool(result.fallback[request_index]),
        "Predictions": prediction_text,
        "Actual Response (ms)": round(
            result.resp_s[request_index] * 1000, 2
        ),
    })

decision_df = pd.DataFrame(decision_rows)

st.dataframe(
    decision_df,
    use_container_width=True,
    hide_index=True,
)
sa_row = decision_df[
    decision_df["Policy"].str.startswith("SA-PLB@")
]

st.subheader("SA-PLB Fallback Verification")

st.write("Decision Branch:", sa_row.iloc[0]["Decision Branch"])
st.write("Fallback Used:", sa_row.iloc[0]["Fallback Used"])

st.info(
    "SA-PLB uses ML when all candidate observations are within "
    "the freshness threshold. Otherwise, it falls back to WRR."
)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("ML Model Information"):

    st.write("Model: HistGradientBoostingRegressor")
    st.write(f"Training samples: {model_metrics['n_train']:,}")
    st.write(f"Validation samples: {model_metrics['n_val']:,}")
    st.write(f"Test MAE: {model_metrics['test_mae_ms']:.2f} ms")

    st.write("Features used:")
    st.write(BASE_FEATURES)


st.caption(
    "Research prototype: all workload, monitoring, interference "
    "and performance measurements are simulated."
)
