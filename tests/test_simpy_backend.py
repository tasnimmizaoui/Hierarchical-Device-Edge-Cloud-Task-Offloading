import pytest

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    FIFOCloudPolicy,
    GreedyNearestDevicePolicy,
    LocalOnlyPolicy,
)
from mec_offloading.domain.actions import DeviceAction
from mec_offloading.domain.models import DeviceState, EdgeState, Task
from mec_offloading.simulation import (
    EdgeConfig,
    SimPyBackend,
    SimulationConfig,
    TopologyConfig,
)
from mec_offloading.workloads import generate_synthetic_tasks


def test_backend_executes_local_edge_and_cloud_routes():
    tasks = generate_synthetic_tasks(count=1, seed=11)
    backend = SimPyBackend()

    local = backend.run_episode(
        tasks, LocalOnlyPolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=11
    )[0]
    edge = backend.run_episode(
        tasks, AlwaysEdgePolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=11
    )[0]
    cloud = backend.run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), EDFCloudPolicy(), seed=11
    )[0]

    assert local.route[-1].value == "device"
    assert edge.route[-1].value == "edge"
    assert cloud.route[-1].value == "cloud"
    assert local.communication_bytes == 0
    assert edge.communication_bytes > 0
    assert cloud.communication_bytes > edge.communication_bytes
    assert cloud.communication_delay_s > edge.communication_delay_s
    assert local.end_to_end_latency_s > 0
    assert edge.device_energy_j > 0
    assert cloud.device_energy_j > 0


def test_external_device_policy_is_called_with_local_observation_only():
    observed: list[tuple[str, float, int]] = []

    class ExternalDevicePolicy:
        def choose(
            self, task: Task, device: DeviceState, edge: EdgeState
        ) -> DeviceAction:
            observed.append((task.task_id, device.battery_fraction, edge.compute.queue_length))
            return DeviceAction.EXECUTE_LOCAL

    task = Task("external-1", "device-1", 0.0, 1.0, 100.0, 0.1, 2.0)
    result = SimPyBackend().run_episode(
        [task],
        ExternalDevicePolicy(),
        EdgeExecutePolicy(),
        EDFCloudPolicy(),
        seed=3,
    )[0]

    assert result.route[-1].value == "device"
    assert observed == [("external-1", 1.0, 0)]


def test_execution_start_time_is_after_queue_wait_at_each_tier():
    tasks = [
        Task("first", "device-1", 0.0, 0.01, 1_000.0, 0.01, 100.0),
        Task("second", "device-1", 0.01, 0.01, 1_000.0, 0.01, 100.0),
    ]

    local = SimPyBackend().run_episode(
        tasks, LocalOnlyPolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=1
    )
    assert local[0].execution_start_time_s == 0.0
    assert local[1].execution_start_time_s >= 0.99

    edge = SimPyBackend().run_episode(
        tasks, AlwaysEdgePolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=1
    )
    assert edge[0].execution_start_time_s > tasks[0].arrival_time_s
    assert edge[1].execution_start_time_s > edge[0].execution_start_time_s

    cloud = SimPyBackend().run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), EDFCloudPolicy(), seed=1
    )
    assert cloud[0].execution_start_time_s > tasks[0].arrival_time_s
    assert cloud[1].execution_start_time_s > cloud[0].execution_start_time_s


def test_greedy_switches_between_local_and_edge_based_on_costs():
    tasks = [
        Task("large-upload", "device-1", 0.0, 20.0, 1_000.0, 0.1, 30.0),
        Task("small-upload", "device-1", 0.1, 0.1, 1_000.0, 0.1, 30.0),
    ]
    backend = SimPyBackend()
    results = backend.run_episode(
        tasks,
        GreedyNearestDevicePolicy(),
        EdgeExecutePolicy(),
        EDFCloudPolicy(),
        seed=17,
    )

    assert [result.route[-1].value for result in results] == ["device", "edge"]


def test_cloud_contention_changes_edf_and_fifo_order_and_deadlines():
    tasks = [
        Task("long-first", "device-1", 0.0, 0.1, 1_000.0, 0.1, 100.0),
        Task("fifo-second", "device-1", 0.05, 0.1, 100.0, 0.1, 100.0),
        Task("urgent-third", "device-1", 0.06, 0.1, 100.0, 0.1, 0.2),
    ]
    config = SimulationConfig(cloud_cpu_capacity_mips=100.0)
    backend = SimPyBackend(config=config)

    fifo = backend.run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), FIFOCloudPolicy(), seed=23
    )
    edf = backend.run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), EDFCloudPolicy(), seed=23
    )

    fifo_order = [r.task_id for r in sorted(fifo, key=lambda r: r.execution_start_time_s)]
    edf_order = [r.task_id for r in sorted(edf, key=lambda r: r.execution_start_time_s)]
    assert fifo_order == ["long-first", "fifo-second", "urgent-third"]
    assert edf_order == ["long-first", "urgent-third", "fifo-second"]
    assert fifo[2].deadline_met is False
    assert edf[2].deadline_met is False


def test_cloud_contention_latency_and_deadline_use_arrival_relative_completion():
    tasks = [
        Task("long-first", "device-1", 0.0, 0.1, 1_000.0, 0.1, 100.0),
        Task("fifo-second", "device-1", 0.05, 0.1, 100.0, 0.1, 100.0),
        Task("urgent-third", "device-1", 0.06, 0.1, 100.0, 0.1, 0.2),
    ]
    results = SimPyBackend(
        config=SimulationConfig(cloud_cpu_capacity_mips=100.0)
    ).run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), FIFOCloudPolicy(), seed=23
    )

    for result, task in zip(results, tasks, strict=True):
        assert result.end_to_end_latency_s == pytest.approx(
            result.completion_time_s - result.arrival_time_s
        )
        assert result.deadline_met is (
            result.completion_time_s <= task.arrival_time_s + task.deadline_s
        )


def _three_device_topology() -> TopologyConfig:
    devices = tuple(
        DeviceState(
            f"device-{index}",
            1_000.0,
            10.0 + index,
            10.0 + index,
            1.5,
            0.5,
            0.001,
        )
        for index in range(1, 4)
    )
    edges = (
        EdgeConfig("edge-1", 1_000.0),
        EdgeConfig("edge-2", 2_000.0),
    )
    return TopologyConfig(
        devices,
        edges,
        (("device-1", "edge-1"), ("device-2", "edge-1"), ("device-3", "edge-2")),
    )


def test_multi_device_multi_edge_topology_preserves_resources_and_origins():
    tasks = [
        Task("d1-cloud", "device-1", 0.0, 0.1, 1_000.0, 0.1, 100.0),
        Task("d2-cloud", "device-2", 0.0, 0.1, 1_000.0, 0.1, 100.0),
        Task("d3-cloud", "device-3", 0.0, 0.1, 1_000.0, 0.1, 100.0),
    ]
    backend = SimPyBackend(
        config=SimulationConfig(
            cloud_cpu_capacity_mips=100.0,
            topology=_three_device_topology(),
        )
    )
    results = backend.run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), EDFCloudPolicy(), seed=4
    )

    assert {(result.source_device_id, result.source_edge_id) for result in results} == {
        ("device-1", "edge-1"),
        ("device-2", "edge-1"),
        ("device-3", "edge-2"),
    }
    assert all(result.route[-1].value == "cloud" for result in results)
    assert all(result.device_energy_j > 0 for result in results)
    assert results[0].communication_bytes > 0


def test_tasks_on_same_edge_share_edge_queue_but_different_edges_do_not():
    tasks = [
        Task("shared-1", "device-1", 0.0, 0.01, 2_000.0, 0.01, 100.0),
        Task("shared-2", "device-2", 0.0, 0.01, 2_000.0, 0.01, 100.0),
        Task("separate", "device-3", 0.0, 0.01, 2_000.0, 0.01, 100.0),
    ]
    results = SimPyBackend(
        config=SimulationConfig(topology=_three_device_topology())
    ).run_episode(
        tasks, AlwaysEdgePolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=9
    )
    by_id = {result.task_id: result for result in results}

    assert by_id["shared-2"].execution_start_time_s > by_id["shared-1"].execution_start_time_s
    assert by_id["separate"].execution_start_time_s < by_id["shared-2"].execution_start_time_s


def test_topology_requires_real_device_associations():
    with pytest.raises(ValueError):
        TopologyConfig(
            (DeviceState("device-1", 1_000.0, 1.0, 1.0, 1.0, 1.0, 0.001),),
            (EdgeConfig("edge-1", 1_000.0),),
            (),
        )


def test_task_with_missing_device_is_rejected():
    task = Task("unknown-device", "device-unknown", 0.0, 0.1, 100.0, 0.1, 10.0)
    with pytest.raises(ValueError, match="unknown device"):
        SimPyBackend().run_episode(
            [task], AlwaysEdgePolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=1
        )
