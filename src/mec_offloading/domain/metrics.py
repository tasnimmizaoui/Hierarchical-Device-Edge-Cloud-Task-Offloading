"""Dimensionally consistent primitive formulas used by all simulator backends."""

from __future__ import annotations

import math


def _non_negative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")


def transmission_time_s(
    data_size_mb: float,
    bandwidth_mbps: float,
    propagation_delay_s: float = 0.0,
    queue_wait_s: float = 0.0,
) -> float:
    """Return transfer duration for decimal MB over Mbps, including delay and queueing.

    1 MB = 8 megabits, therefore serialization time = (MB * 8) / Mbps.
    """
    _non_negative_finite("data_size_mb", data_size_mb)
    _non_negative_finite("propagation_delay_s", propagation_delay_s)
    _non_negative_finite("queue_wait_s", queue_wait_s)
    if not math.isfinite(bandwidth_mbps) or bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be finite and > 0")
    return (data_size_mb * 8.0 / bandwidth_mbps) + propagation_delay_s + queue_wait_s


def execution_time_s(
    required_mi: float,
    capacity_mips: float,
    queue_wait_s: float = 0.0,
) -> float:
    """Return compute duration plus queue wait (MI / MIPS = seconds)."""
    if not math.isfinite(required_mi) or required_mi <= 0:
        raise ValueError("required_mi must be finite and > 0")
    if not math.isfinite(capacity_mips) or capacity_mips <= 0:
        raise ValueError("capacity_mips must be finite and > 0")
    _non_negative_finite("queue_wait_s", queue_wait_s)
    return required_mi / capacity_mips + queue_wait_s


def deadline_met(completion_time_s: float, arrival_time_s: float, deadline_s: float) -> bool:
    """Whether a completion timestamp meets a relative deadline from task arrival."""
    _non_negative_finite("completion_time_s", completion_time_s)
    _non_negative_finite("arrival_time_s", arrival_time_s)
    if not math.isfinite(deadline_s) or deadline_s <= 0:
        raise ValueError("deadline_s must be finite and > 0")
    return completion_time_s <= arrival_time_s + deadline_s


def radio_energy_j(tx_power_w: float, transmission_duration_s: float) -> float:
    """First-order transmit energy estimate: power (W) × duration (s) = joules."""
    _non_negative_finite("tx_power_w", tx_power_w)
    _non_negative_finite("transmission_duration_s", transmission_duration_s)
    return tx_power_w * transmission_duration_s


def cpu_energy_j(cpu_energy_per_mi_j: float, required_mi: float) -> float:
    """Simple device CPU energy model; calibrate/justify coefficient in experiments."""
    _non_negative_finite("cpu_energy_per_mi_j", cpu_energy_per_mi_j)
    if not math.isfinite(required_mi) or required_mi <= 0:
        raise ValueError("required_mi must be finite and > 0")
    return cpu_energy_per_mi_j * required_mi
