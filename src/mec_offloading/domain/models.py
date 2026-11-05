"""Shared domain models. Internal units follow docs/interfaces.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):  # noqa: UP042
    DEVICE = "device"
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass(frozen=True, slots=True)
class Task:
    """Independent compute task; deadline_s is relative to arrival_time_s."""

    task_id: str
    device_id: str
    arrival_time_s: float
    input_size_mb: float
    required_mi: float
    result_size_mb: float
    deadline_s: float
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        for field_name in (
            "arrival_time_s",
            "input_size_mb",
            "required_mi",
            "result_size_mb",
            "deadline_s",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if self.required_mi <= 0:
            raise ValueError("required_mi must be > 0")
        if self.deadline_s <= 0:
            raise ValueError("deadline_s must be > 0")

    @property
    def absolute_deadline_s(self) -> float:
        """Absolute simulation timestamp by which the task must complete."""
        return self.arrival_time_s + self.deadline_s


@dataclass(frozen=True, slots=True)
class ComputeNodeState:
    """Observable compute-node state supplied to a policy."""

    node_id: str
    tier: Tier
    cpu_capacity_mips: float
    cpu_utilization: float = 0.0
    queue_wait_s: float = 0.0
    queue_length: int = 0

    def __post_init__(self) -> None:
        if self.cpu_capacity_mips <= 0:
            raise ValueError("cpu_capacity_mips must be > 0")
        if not 0.0 <= self.cpu_utilization <= 1.0:
            raise ValueError("cpu_utilization must be between 0 and 1")
        if self.queue_wait_s < 0 or self.queue_length < 0:
            raise ValueError("queue metrics must be >= 0")


@dataclass(frozen=True, slots=True)
class DeviceState:
    """Device compute, battery and radio state."""

    device_id: str
    cpu_capacity_mips: float
    battery_j: float
    battery_capacity_j: float
    tx_power_w: float
    rx_power_w: float
    cpu_energy_per_mi_j: float

    def __post_init__(self) -> None:
        if self.cpu_capacity_mips <= 0 or self.battery_capacity_j <= 0:
            raise ValueError("device CPU and battery capacity must be > 0")
        if not 0 <= self.battery_j <= self.battery_capacity_j:
            raise ValueError("battery_j must be in [0, battery_capacity_j]")
        for field_name in ("tx_power_w", "rx_power_w", "cpu_energy_per_mi_j"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")

    @property
    def battery_fraction(self) -> float:
        return self.battery_j / self.battery_capacity_j


@dataclass(frozen=True, slots=True)
class NetworkLinkState:
    """One directed link; bandwidth is Mbps and propagation delay is seconds."""

    bandwidth_mbps: float
    propagation_delay_s: float
    queue_wait_s: float = 0.0
    utilization: float = 0.0

    def __post_init__(self) -> None:
        if self.bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be > 0")
        if self.propagation_delay_s < 0 or self.queue_wait_s < 0:
            raise ValueError("network delays must be >= 0")
        if not 0 <= self.utilization <= 1:
            raise ValueError("utilization must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class EdgeState:
    """Edge-side state, including the two links used by the hierarchy."""

    compute: ComputeNodeState
    device_edge_link: NetworkLinkState
    edge_cloud_link: NetworkLinkState


@dataclass(frozen=True, slots=True)
class CloudState:
    """Cloud/HPC compute and queue summary."""

    compute: ComputeNodeState


@dataclass(frozen=True, slots=True)
class TaskResult:
    """End-to-end outcome for a completed or failed task."""

    task_id: str
    route: tuple[Tier, ...]
    arrival_time_s: float
    completion_time_s: float
    end_to_end_latency_s: float
    deadline_met: bool
    device_energy_j: float
    communication_delay_s: float
    communication_bytes: int
    failure_reason: str | None = None
    execution_start_time_s: float = 0.0
    feedback_messages: int = 0
    feedback_bytes: int = 0
    feedback_modeled_delay_s: float = 0.0


@dataclass(frozen=True, slots=True)
class QueueSummary:
    """Fixed-shape summary passed to the cloud scheduling policy."""

    queue_length: int
    min_remaining_deadline_s: float
    mean_remaining_deadline_s: float
    mean_execution_time_s: float
    oldest_wait_s: float
