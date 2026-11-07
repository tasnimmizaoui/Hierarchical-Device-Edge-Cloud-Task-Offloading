import pytest

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    DelayedGreedyComputeEdgePolicy,
    DelayedGreedyNearestDevicePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    FIFOCloudPolicy,
    GreedyNearestDevicePolicy,
    LocalOnlyPolicy,
)
from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.actions import (
    CloudSchedulingRule,
    DeviceAction,
    EdgeAction,
)
from mec_offloading.domain.models import DeviceState, EdgeState, Task
from mec_offloading.rl.policies import CloudDQNPolicy
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


def test_queue_wait_metric_includes_time_waiting_for_compute():
    tasks = [
        Task("first", "device-1", 0.0, .01, 1_000.0, .01, 100.0),
        Task("second", "device-1", .1, .01, 1_000.0, .01, 100.0),
    ]
    results = SimPyBackend(
        config=SimulationConfig(device_cpu_capacity_mips=100.0)
    ).run_episode(
        tasks, LocalOnlyPolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=2
    )
    assert results[0].queue_wait_s == 0
    assert results[1].queue_wait_s == pytest.approx(9.9)
    assert results[1].queue_wait_s > 1


def test_seeded_bandwidth_and_service_noise_are_reproducible():
    task = Task("noisy", "device-1", 0.0, 1.0, 1_000.0, .1, 100.0)
    backend = SimPyBackend(config=SimulationConfig(
        bandwidth_jitter_fraction=.2,
        service_time_noise_fraction=.2,
    ))

    def run(seed):
        return backend.run_episode(
            [task], AlwaysEdgePolicy(), EdgeExecutePolicy(), EDFCloudPolicy(),
            seed=seed,
        )[0]

    assert run(7) == run(7)
    assert run(7).end_to_end_latency_s != run(8).end_to_end_latency_s


def test_delayed_greedy_baseline_runs_with_causal_feedback():
    tasks = [
        Task("delayed-1", "device-1", 0.0, .1, 100.0, .1, 10.0),
        Task("delayed-2", "device-2", .01, .1, 100.0, .1, 10.0),
    ]
    coordinator = FeedbackCoordinator("frequent")
    results = SimPyBackend(
        config=SimulationConfig(topology=_three_device_topology())
    ).run_episode(
        tasks,
        DelayedGreedyNearestDevicePolicy(coordinator),
        DelayedGreedyComputeEdgePolicy(coordinator),
        EDFCloudPolicy(),
        seed=6,
    )
    assert len(results) == 2
    assert all(result.feedback_messages > 0 for result in results)


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


def test_spt_schedules_by_estimated_work_not_simulated_work():
    tasks = [
        Task("first", "device-1", 0.0, .1, 1_000.0, .1, 100.0),
        Task("actual-short", "device-1", .05, .1, 100.0, .1, 100.0,
             estimated_required_mi=900.0),
        Task("estimated-short", "device-1", .06, .1, 200.0, .1, 100.0,
             estimated_required_mi=50.0),
    ]

    class FixedCloudPolicy:
        def choose_rule(self, queue, cloud):
            del queue, cloud
            return CloudSchedulingRule.SHORTEST_PROCESSING_TIME

    results = SimPyBackend(
        config=SimulationConfig(cloud_cpu_capacity_mips=100.0)
    ).run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), FixedCloudPolicy(), seed=4
    )
    order = [
        result.task_id
        for result in sorted(results, key=lambda result: result.execution_start_time_s)
    ]
    assert order == ["first", "estimated-short", "actual-short"]


def test_singleton_cloud_rules_match_and_dqn_forced_to_spt_matches_fixed_spt():
    task = Task("only", "device-1", 0.0, .1, 100.0, .1, 100.0)

    class FixedActionAgent:
        def act(self, observation, training=False):
            del observation, training
            return int(CloudSchedulingRule.SHORTEST_PROCESSING_TIME)

    class FixedCloudPolicy:
        def choose_rule(self, queue, cloud):
            del queue, cloud
            return CloudSchedulingRule.SHORTEST_PROCESSING_TIME

    backend = SimPyBackend()
    single_results = []
    for selected_rule in CloudSchedulingRule:
        class OneRule:
            def choose_rule(self, queue, cloud, rule=selected_rule):
                del queue, cloud
                return rule

        single_results.append(backend.run_episode(
            [task], AlwaysEdgePolicy(), EdgeForwardPolicy(), OneRule(), seed=8
        )[0])
    assert single_results[0] == single_results[1] == single_results[2]

    tasks = [
        Task("first", "device-1", 0.0, .1, 1_000.0, .1, 100.0),
        Task("short", "device-1", .05, .1, 100.0, .1, 100.0),
        Task("medium", "device-1", .06, .1, 200.0, .1, 100.0),
    ]
    contested_backend = SimPyBackend(config=SimulationConfig(
        edge_cpu_capacity_mips=1_000_000.0,
        cloud_cpu_capacity_mips=100.0,
    ))
    learned_policy = CloudDQNPolicy(FixedActionAgent())
    learned = contested_backend.run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), learned_policy, seed=8
    )
    fixed = contested_backend.run_episode(
        tasks, AlwaysEdgePolicy(), EdgeForwardPolicy(), FixedCloudPolicy(), seed=8
    )
    assert learned == fixed
    assert len(learned_policy.log.entries) == 1


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


def test_compute_queue_wait_estimates_remaining_and_queued_work():
    tasks = [
        Task("edge-active", "device-1", 0.0, 0.01, 5_000.0, 0.01, 100.0),
        Task("edge-next", "device-2", 0.5, 0.01, 1_000.0, 0.01, 100.0),
    ]

    class CaptureEdgeWait:
        def __init__(self):
            self.wait_by_task = {}

        def choose(self, task, edge, cloud):
            del cloud
            self.wait_by_task[task.task_id] = edge.compute.queue_wait_s
            return EdgeAction.EXECUTE_AT_EDGE

    edge_policy = CaptureEdgeWait()
    SimPyBackend(config=SimulationConfig(topology=_three_device_topology())).run_episode(
        tasks, AlwaysEdgePolicy(), edge_policy, EDFCloudPolicy(), seed=10
    )
    assert edge_policy.wait_by_task["edge-next"] > 4.0

    class CaptureCloudWait:
        def __init__(self):
            self.wait_by_task = {}

        def choose(self, task, edge, cloud):
            del edge
            self.wait_by_task[task.task_id] = cloud.compute.queue_wait_s
            return EdgeAction.FORWARD_TO_CLOUD

    cloud_policy = CaptureCloudWait()
    SimPyBackend(
        config=SimulationConfig(
            cloud_cpu_capacity_mips=1_000.0,
            topology=_three_device_topology(),
        )
    ).run_episode(
        tasks, AlwaysEdgePolicy(), cloud_policy, EDFCloudPolicy(), seed=10
    )
    assert cloud_policy.wait_by_task["edge-next"] > 4.0


def test_device_policy_observes_its_active_and_queued_compute_work():
    tasks = [
        Task("device-active", "device-1", 0.0, 0.01, 5_000.0, 0.01, 100.0),
        Task("device-next", "device-1", 0.5, 0.01, 1_000.0, 0.01, 100.0),
    ]

    class CaptureDeviceQueue:
        def __init__(self):
            self.wait_by_task = {}

        def choose(self, task, device, edge):
            del edge
            self.wait_by_task[task.task_id] = device.queue_wait_s
            return DeviceAction.EXECUTE_LOCAL

    device_policy = CaptureDeviceQueue()
    SimPyBackend().run_episode(
        tasks, device_policy, EdgeExecutePolicy(), EDFCloudPolicy(), seed=10
    )
    assert device_policy.wait_by_task["device-next"] > 4.0


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
