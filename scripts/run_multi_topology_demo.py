"""Demonstrate the configurable multi-device, multi-edge SimPy topology."""

from __future__ import annotations

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
)
from mec_offloading.domain.actions import EdgeAction
from mec_offloading.domain.models import CloudState, DeviceState, EdgeState, Task
from mec_offloading.simulation import EdgeConfig, SimPyBackend, SimulationConfig, TopologyConfig


class DemonstrationEdgePolicy:
    """Execute selected tasks at the edge and forward the others to the cloud."""

    def __init__(self, cloud_task_ids: set[str]) -> None:
        self._cloud_task_ids = cloud_task_ids

    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        del edge, cloud
        return (
            EdgeAction.FORWARD_TO_CLOUD
            if task.task_id in self._cloud_task_ids
            else EdgeAction.EXECUTE_AT_EDGE
        )


def build_topology() -> TopologyConfig:
    devices = (
        DeviceState("device-1", 1_000.0, 100.0, 100.0, 1.5, 0.5, 0.001),
        DeviceState("device-2", 800.0, 80.0, 80.0, 1.2, 0.4, 0.001),
        DeviceState("device-3", 1_200.0, 120.0, 120.0, 1.8, 0.6, 0.001),
    )
    edges = (
        EdgeConfig("edge-1", 500.0),
        EdgeConfig("edge-2", 750.0),
    )
    return TopologyConfig(
        devices=devices,
        edges=edges,
        device_to_edge=(
            ("device-1", "edge-1"),
            ("device-2", "edge-1"),
            ("device-3", "edge-2"),
        ),
    )


def build_tasks() -> list[Task]:
    return [
        Task("edge1-device1", "device-1", 0.00, 0.2, 1_000.0, 0.1, 100.0),
        Task("edge1-device2", "device-2", 0.00, 0.2, 1_000.0, 0.1, 100.0),
        Task("edge2-device3", "device-3", 0.00, 0.2, 1_000.0, 0.1, 100.0),
        Task("cloud-edge1", "device-1", 0.20, 0.2, 2_000.0, 0.1, 10.0),
        Task("cloud-edge2", "device-3", 0.20, 0.2, 1_000.0, 0.1, 1.0),
    ]


def print_results(results: list) -> None:
    headers = (
        "Task ID",
        "Device ID",
        "Selected Edge",
        "Route",
        "Arrival",
        "Completion",
        "Latency",
        "Device Energy",
        "Deadline",
    )
    rows = [
        (
            result.task_id,
            result.source_device_id,
            result.source_edge_id or "-",
            "->".join(tier.value for tier in result.route),
            f"{result.arrival_time_s:.4f}",
            f"{result.completion_time_s:.4f}",
            f"{result.end_to_end_latency_s:.4f}",
            f"{result.device_energy_j:.4f}",
            "yes" if result.deadline_met else "no",
        )
        for result in results
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    tasks = build_tasks()
    cloud_task_ids = {"cloud-edge1", "cloud-edge2"}
    backend = SimPyBackend(
        config=SimulationConfig(
            cloud_cpu_capacity_mips=500.0,
            topology=build_topology(),
        )
    )
    results = backend.run_episode(
        tasks,
        AlwaysEdgePolicy(),
        DemonstrationEdgePolicy(cloud_task_ids),
        EDFCloudPolicy(),
        seed=2026,
    )

    print("Multi-device, multi-edge SimPy demonstration")
    print("Topology: device-1/device-2 -> edge-1; device-3 -> edge-2; shared cloud")
    print("Devices: device-1=1000 MIPS/100 J, device-2=800 MIPS/80 J, device-3=1200 MIPS/120 J")
    print("Edges: edge-1=500 MIPS, edge-2=750 MIPS; cloud=500 MIPS")
    print()
    print_results(results)
    print()
    print("Cloud execution order:")
    cloud_results = sorted(
        (result for result in results if result.task_id in cloud_task_ids),
        key=lambda result: result.execution_start_time_s,
    )
    for index, result in enumerate(cloud_results, start=1):
        print(
            f"  {index}. {result.task_id} "
            f"(device={result.source_device_id}, edge={result.source_edge_id}, "
            f"start={result.execution_start_time_s:.4f}s)"
        )
    print()
    print("Energy by device:")
    for device_id in ("device-1", "device-2", "device-3"):
        energy = sum(
            result.device_energy_j
            for result in results
            if result.source_device_id == device_id
        )
        print(f"  {device_id}: {energy:.4f} J")


if __name__ == "__main__":
    main()
