from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    LocalOnlyPolicy,
)
from mec_offloading.domain.actions import DeviceAction
from mec_offloading.domain.models import DeviceState, EdgeState, Task
from mec_offloading.simulation import SimPyBackend
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
