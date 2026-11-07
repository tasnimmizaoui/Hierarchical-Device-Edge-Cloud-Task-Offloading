"""Frozen S2 observation/action-space registry; optional Gymnasium adapters.

A registry is not a functioning Gymnasium multi-agent environment. Student 1
must first expose a stepable simulator protocol for reset/step on one timeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgentSpec:
    name: str
    observation_features: tuple[str, ...]
    actions: tuple[str, ...]

    @property
    def observation_dim(self) -> int:
        return len(self.observation_features)

    def validate(self, observation: np.ndarray, action: int | None = None) -> None:
        x = np.asarray(observation)
        if x.shape != (self.observation_dim,):
            raise ValueError(f"{self.name}: expected {self.observation_dim} features")
        if not np.isfinite(x).all():
            raise ValueError(f"{self.name}: non-finite observation")
        if action is not None and action not in range(len(self.actions)):
            raise ValueError(f"{self.name}: invalid action {action}")

    def gym_spaces(self):
        """Return optional gymnasium spaces; no simulator imported."""
        try:
            from gymnasium import spaces
        except ImportError as exc:
            raise RuntimeError("Install gymnasium: pip install -e '.[dev,rl]'") from exc
        return (spaces.Box(low=-np.inf, high=np.inf,
                           shape=(self.observation_dim,), dtype=np.float32),
                spaces.Discrete(len(self.actions)))


SPECS = {
    "device": AgentSpec("device", (
        "work_log_scaled", "input_scaled_mb", "deadline_to_local_service_ratio",
        "battery_fraction", "device_cpu_scaled_mips", "local_queue_delay_log_scaled_s",
        "offload_deadline_service_ratio",
        "device_edge_bw_scaled_mbps", "device_edge_propagation_scaled_s",
        "edge_congestion_signal", "has_edge_signal",
    ), ("EXECUTE_LOCAL", "OFFLOAD_TO_EDGE")),
    "edge": AgentSpec("edge", (
        "edge_queue_log_scaled", "edge_util", "edge_cpu_scaled_mips",
        "edge_queue_delay_log_scaled_s", "deadline_to_edge_service_ratio",
        "work_log_scaled",
        "edge_cloud_bw_scaled_mbps", "edge_cloud_propagation_scaled_s",
        "cloud_congestion_signal", "has_cloud_signal",
    ), ("EXECUTE_EDGE", "FORWARD_CLOUD")),
    "cloud": AgentSpec("cloud", (
        "queue_log_scaled", "cloud_util", "cloud_cpu_scaled_mips",
        "cloud_queue_delay_log_scaled_s", "min_deadline_slack_log_scaled_s",
        "mean_deadline_slack_log_scaled_s", "mean_execution_log_scaled_s",
        "oldest_wait_log_scaled_s", "queue_nonempty",
        "near_deadline_queue_log_scaled", "workload_imbalance_log_scaled",
    ), ("EDF", "SPT", "FIFO")),
    "centralized": AgentSpec("centralized", (
        "work_log_scaled", "input_scaled_mb", "deadline_to_local_service_ratio",
        "battery_fraction", "device_cpu_scaled_mips", "local_queue_delay_log_scaled_s",
        "offload_deadline_service_ratio",
        "edge_cpu_scaled_mips", "edge_queue_delay_log_scaled_s", "edge_util",
        "cloud_cpu_scaled_mips", "cloud_queue_log_scaled", "cloud_util",
        "device_edge_bw_scaled_mbps", "edge_cloud_bw_scaled_mbps",
        "cloud_queue_delay_scaled_s",
    ), ("DEVICE", "EDGE", "CLOUD")),
}
