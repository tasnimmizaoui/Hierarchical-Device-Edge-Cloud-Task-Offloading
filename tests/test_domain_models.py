import pytest

from mec_offloading.domain.models import (
    ComputeNodeState,
    DeviceState,
    NetworkLinkState,
    Task,
    Tier,
)


def test_task_absolute_deadline_is_arrival_plus_relative_deadline():
    task = Task(
        task_id="task-1",
        device_id="device-1",
        arrival_time_s=12.0,
        input_size_mb=2.5,
        required_mi=100.0,
        result_size_mb=0.2,
        deadline_s=3.0,
    )
    assert task.absolute_deadline_s == 15.0


def test_task_rejects_non_positive_compute_demand():
    with pytest.raises(ValueError, match="required_mi"):
        Task("t", "d", 0.0, 1.0, 0.0, 0.1, 2.0)


def test_device_battery_fraction():
    device = DeviceState("d1", 500.0, 25.0, 100.0, 1.5, 0.5, 0.001)
    assert device.battery_fraction == 0.25


def test_compute_utilization_must_be_a_fraction():
    with pytest.raises(ValueError, match="cpu_utilization"):
        ComputeNodeState("edge-1", Tier.EDGE, 10_000.0, 1.2)


def test_network_bandwidth_must_be_positive():
    with pytest.raises(ValueError, match="bandwidth_mbps"):
        NetworkLinkState(0.0, 0.01)
