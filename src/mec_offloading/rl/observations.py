"""Canonical SimPy-domain -> normalized DQN feature vectors.

The feature ordering and feedback semantics are version 3. In particular these observations are NOT
compatible with checkpoints trained on the standalone RL.zip surrogate.
"""
from __future__ import annotations

import numpy as np

from mec_offloading.domain.models import (
    CloudState,
    DeviceState,
    EdgeState,
    QueueSummary,
    Task,
)

FEATURE_VERSION = "simpy-v9-cloud-queue-signals"

_MAX_WORK_MI = 100_000_000.0
_MAX_TIME_S = 10_000_000.0
_MAX_QUEUE = 1_000.0


def _workload_scale(required_mi: float) -> float:
    """Keep Alibaba's heavy-tailed MI values in a stable, ordered range."""
    return min(1.0, np.log1p(required_mi) / np.log1p(_MAX_WORK_MI))


def _deadline_service_ratio(remaining_s: float, required_mi: float,
                            capacity_mips: float) -> float:
    service_s = max(required_mi / capacity_mips, 1e-6)
    return _remaining_service_ratio(remaining_s, service_s)


def _remaining_service_ratio(remaining_s: float, service_s: float) -> float:
    service_s = max(service_s, 1e-6)
    ratio = remaining_s / service_s
    return float(ratio / (1.0 + abs(ratio)))


def _edge_route_service_s(task: Task, edge: EdgeState) -> float:
    upload_s = (
        task.input_size_mb * 8.0 / edge.device_edge_link.bandwidth_mbps
        + edge.device_edge_link.propagation_delay_s
    )
    download_s = (
        task.result_size_mb * 8.0 / edge.device_edge_link.bandwidth_mbps
        + edge.device_edge_link.propagation_delay_s
    )
    return (
        upload_s
        + edge.compute.queue_wait_s
        + task.required_mi / edge.compute.cpu_capacity_mips
        + download_s
    )


def _signed_time_scale(seconds: float) -> float:
    magnitude = min(
        1.0,
        np.log1p(abs(seconds)) / np.log1p(_MAX_TIME_S),
    )
    return float(np.copysign(magnitude, seconds))


def _queue_scale(queue_length: int) -> float:
    return min(1.0, np.log1p(queue_length) / np.log1p(_MAX_QUEUE))


def _linear_scale(value: float, reference: float) -> float:
    return min(1.0, value / reference)


def device_observation(task: Task, device: DeviceState, edge: EdgeState,
                       now_s: float, edge_signal: float | None) -> np.ndarray:
    link = edge.device_edge_link
    return np.asarray([
        _workload_scale(task.required_mi),
        _linear_scale(task.input_size_mb, 8.0),
        _deadline_service_ratio(
            task.absolute_deadline_s - now_s,
            task.required_mi + device.queue_wait_s * device.cpu_capacity_mips,
            device.cpu_capacity_mips,
        ),
        device.battery_fraction,
        _linear_scale(device.cpu_capacity_mips, 1200.0),
        _signed_time_scale(device.queue_wait_s),
        _remaining_service_ratio(
            task.absolute_deadline_s - now_s,
            _edge_route_service_s(task, edge),
        ),
        _linear_scale(link.bandwidth_mbps, 100.0),
        _linear_scale(link.propagation_delay_s, 0.05),
        -1.0 if edge_signal is None else edge_signal,
        float(edge_signal is not None),
    ], dtype=np.float32)


def edge_observation(task: Task, edge: EdgeState, now_s: float,
                     cloud_signal: float | None) -> np.ndarray:
    compute, link = edge.compute, edge.edge_cloud_link
    return np.asarray([
        _queue_scale(compute.queue_length),
        compute.cpu_utilization,
        _linear_scale(compute.cpu_capacity_mips, 7500.0),
        _signed_time_scale(compute.queue_wait_s),
        _deadline_service_ratio(
            task.absolute_deadline_s - now_s, task.required_mi,
            compute.cpu_capacity_mips,
        ),
        _workload_scale(task.required_mi),
        _linear_scale(link.bandwidth_mbps, 1000.0),
        _linear_scale(link.propagation_delay_s, 0.06),
        -1.0 if cloud_signal is None else cloud_signal,
        float(cloud_signal is not None),
    ], dtype=np.float32)


def cloud_observation(queue: QueueSummary, cloud: CloudState) -> np.ndarray:
    c = cloud.compute
    workload_ratio = min(1.0, np.log1p(max(1.0, queue.workload_ratio)) / np.log1p(1_000.0))
    return np.asarray([
        _queue_scale(queue.queue_length),
        c.cpu_utilization,
        _linear_scale(c.cpu_capacity_mips, 20000.0),
        _signed_time_scale(c.queue_wait_s),
        _signed_time_scale(queue.min_remaining_deadline_s),
        _signed_time_scale(queue.mean_remaining_deadline_s),
        _signed_time_scale(queue.mean_execution_time_s),
        _signed_time_scale(queue.oldest_wait_s),
        1.0 if queue.queue_length > 0 else 0.0,
        _queue_scale(queue.near_deadline_count),
        float(workload_ratio),
    ], dtype=np.float32)


def centralized_observation(task: Task, device: DeviceState, edge: EdgeState,
                            cloud: CloudState, now_s: float) -> np.ndarray:
    """Full-state oracle is ONLY used in the centralized RL reference."""
    return np.asarray([
        _workload_scale(task.required_mi),
        _linear_scale(task.input_size_mb, 8.0),
        _deadline_service_ratio(
            task.absolute_deadline_s - now_s,
            task.required_mi + device.queue_wait_s * device.cpu_capacity_mips,
            device.cpu_capacity_mips,
        ),
        device.battery_fraction,
        _linear_scale(device.cpu_capacity_mips, 1200.0),
        _signed_time_scale(device.queue_wait_s),
        _remaining_service_ratio(
            task.absolute_deadline_s - now_s,
            _edge_route_service_s(task, edge),
        ),
        _linear_scale(edge.compute.cpu_capacity_mips, 7500.0),
        _signed_time_scale(edge.compute.queue_wait_s),
        edge.compute.cpu_utilization,
        _linear_scale(cloud.compute.cpu_capacity_mips, 20000.0),
        _queue_scale(cloud.compute.queue_length),
        cloud.compute.cpu_utilization,
        _linear_scale(edge.device_edge_link.bandwidth_mbps, 100.0),
        _linear_scale(edge.edge_cloud_link.bandwidth_mbps, 1000.0),
        _signed_time_scale(cloud.compute.queue_wait_s),
    ], dtype=np.float32)
