"""All simulation assumptions live here so they are documented in one place."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ReplicaSpec:
    id: int
    cores: int        # parallel servers in the replica
    speed: float      # relative speed (higher = faster)
    mem_mb: int


DEFAULT_REPLICAS: Tuple[ReplicaSpec, ...] = (
    ReplicaSpec(0, 2, 1.00, 1024),
    ReplicaSpec(1, 2, 0.90, 1024),
    ReplicaSpec(2, 1, 1.10, 512),
)


@dataclass(frozen=True)
class SimConfig:
    replicas: Tuple[ReplicaSpec, ...] = DEFAULT_REPLICAS

    # --- request / service model (SIMULATED) ---
    base_service_s: float = 0.12          # fixed work per request
    per_kb_service_s: float = 0.00025     # extra work per KB of payload
    size_log_mean: float = 5.5214609      # ln(250) KB
    size_log_sigma: float = 0.9
    size_min_kb: float = 5.0
    size_max_kb: float = 5000.0
    type_names: Tuple[str, ...] = ("GET", "POST", "PUT")
    type_prob: Tuple[float, ...] = (0.60, 0.30, 0.10)
    type_cost: Tuple[float, ...] = (1.0, 1.6, 1.3)
    service_noise_sigma: float = 0.15     # log-normal noise on each (request, replica) service time

    # --- monitoring model (SIMULATED) ---
    monitor_interval_s: float = 0.5       # nominal sampling period of each replica's monitor
    monitor_jitter_frac: float = 0.10     # +/- fraction jitter on the period
    monitor_base_delay_s: float = 0.05    # minimum collection -> delivery delay
    monitor_delay_mean_s: float = 0.1     # mean of extra exponential delivery delay
    stall_prob: float = 0.03              # chance an observation is additionally stalled
    stall_extra_s: float = 5.0            # extra delay when stalled
    recent_window: int = 50               # "recent average response time" = last N completions
    mem_base_pct: float = 20.0
    mem_per_active_mb: float = 9.0
    cpu_noise_sd: float = 2.0
    mem_noise_sd: float = 0.7

    # --- routing ---
    wrr_weights: Optional[Tuple[int, ...]] = None   # None -> round(2*cores*speed)

    def weights(self) -> Tuple[int, ...]:
        if self.wrr_weights is not None:
            return self.wrr_weights
        return tuple(max(1, round(2 * r.cores * r.speed)) for r in self.replicas)

    @property
    def n_replicas(self) -> int:
        return len(self.replicas)
