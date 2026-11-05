"""Canonical SimPy-domain -> normalized DQN feature vectors.

The feature ordering and feedback semantics are version 3. In particular these observations are NOT
compatible with checkpoints trained on the standalone RL.zip surrogate.
"""
from __future__ import annotations

import numpy as np

from mec_offloading.domain.models import (
    CloudState, DeviceState, EdgeState, QueueSummary, Task,
)

FEATURE_VERSION = "simpy-v3-causal-feedback"


def device_observation(task: Task, device: DeviceState, edge: EdgeState,
                       now_s: float, edge_signal: float | None) -> np.ndarray:
    link = edge.device_edge_link
    # Current local-device CPU queue is not supplied by Interface v1: -1 = unknown.
    return np.asarray([
        task.required_mi / 3500.0,
        task.input_size_mb / 8.0,
        (task.absolute_deadline_s - now_s) / 12.0,
        device.battery_fraction,
        device.cpu_capacity_mips / 1100.0,
        -1.0,
        link.bandwidth_mbps / 72.0,
        link.propagation_delay_s / 0.05,
        -1.0 if edge_signal is None else edge_signal,
        float(edge_signal is not None),
    ], dtype=np.float32)


def edge_observation(task: Task, edge: EdgeState, now_s: float,
                     cloud_signal: float | None) -> np.ndarray:
    compute, link = edge.compute, edge.edge_cloud_link
    return np.asarray([
        compute.queue_length / 15.0,
        compute.cpu_utilization,
        compute.cpu_capacity_mips / 5200.0,
        compute.queue_wait_s / 3.0,
        (task.absolute_deadline_s - now_s) / 12.0,
        task.required_mi / 3500.0,
        link.bandwidth_mbps / 336.0,
        link.propagation_delay_s / 0.06,
        -1.0 if cloud_signal is None else cloud_signal,
        float(cloud_signal is not None),
    ], dtype=np.float32)


def cloud_observation(queue: QueueSummary, cloud: CloudState) -> np.ndarray:
    c = cloud.compute
    return np.asarray([
        queue.queue_length / 15.0,
        c.cpu_utilization,
        c.cpu_capacity_mips / 18000.0,
        c.queue_wait_s / 3.0,
        queue.min_remaining_deadline_s / 12.0,
        queue.mean_remaining_deadline_s / 12.0,
        queue.mean_execution_time_s / 3.0,
        queue.oldest_wait_s / 12.0,
        1.0 if queue.queue_length > 0 else 0.0,
    ], dtype=np.float32)


def centralized_observation(task: Task, device: DeviceState, edge: EdgeState,
                            cloud: CloudState, now_s: float) -> np.ndarray:
    """Full-state oracle is ONLY used in the centralized RL reference."""
    return np.asarray([
        task.required_mi / 3500.0,
        task.input_size_mb / 8.0,
        (task.absolute_deadline_s - now_s) / 12.0,
        device.battery_fraction,
        device.cpu_capacity_mips / 1100.0,
        -1.0,  # local queue is unavailable in interface v1
        edge.compute.cpu_capacity_mips / 5200.0,
        edge.compute.queue_wait_s / 3.0,
        edge.compute.cpu_utilization,
        cloud.compute.cpu_capacity_mips / 18000.0,
        cloud.compute.queue_length / 15.0,
        cloud.compute.cpu_utilization,
        edge.device_edge_link.bandwidth_mbps / 72.0,
        edge.edge_cloud_link.bandwidth_mbps / 336.0,
        cloud.compute.queue_wait_s / 3.0,
    ], dtype=np.float32)
