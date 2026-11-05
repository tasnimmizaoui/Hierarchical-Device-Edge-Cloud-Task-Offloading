"""Run the controlled greedy-switch and cloud-contention scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    FIFOCloudPolicy,
    GreedyNearestDevicePolicy,
)
from mec_offloading.domain.models import Task
from mec_offloading.simulation import SimPyBackend, SimulationConfig


def load_scenario(path: Path) -> tuple[dict, list[Task]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    workload = data["workload"]["tasks"]
    tasks = [
        Task(device_id="device-1", priority=0, **task_data)
        for task_data in workload
    ]
    resources = data["resources"]
    network = data["network"]
    data["conditions"] = data.get("conditions", [{}])
    for condition in data["conditions"]:
        condition.setdefault("device_edge_bandwidth_mbps", network["device_edge_bandwidth_mbps"])
        condition.setdefault("edge_cpu_mips", resources["edge_cpu_mips"])
    return data, tasks


def build_config(data: dict, condition: dict) -> SimulationConfig:
    resources = data["resources"]
    network = data["network"]
    config = SimulationConfig(
        device_cpu_capacity_mips=resources["device_cpu_mips"],
        edge_cpu_capacity_mips=condition["edge_cpu_mips"],
        cloud_cpu_capacity_mips=resources["cloud_cpu_mips"],
        device_edge_bandwidth_mbps=condition["device_edge_bandwidth_mbps"],
        edge_cloud_bandwidth_mbps=network["edge_cloud_bandwidth_mbps"],
        device_edge_propagation_s=network["device_edge_propagation_ms"] / 1000,
        edge_cloud_propagation_s=network["edge_cloud_propagation_ms"] / 1000,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    data, tasks = load_scenario(args.scenario)
    name = data["scenario"]["name"]

    if name == "greedy-switch":
        print(name)
        for condition in data["conditions"]:
            results = SimPyBackend(config=build_config(data, condition)).run_episode(
                tasks,
                GreedyNearestDevicePolicy(),
                EdgeExecutePolicy(),
                EDFCloudPolicy(),
                seed=data["scenario"]["seed"],
            )
            print(condition["name"])
            for result in results:
                print(
                    result.task_id,
                    result.route[-1].value,
                    f"latency={result.end_to_end_latency_s:.6f}s",
                )
        return

    backend = SimPyBackend(config=build_config(data, data["conditions"][0]))
    for label, cloud_policy in (("fifo", FIFOCloudPolicy()), ("edf", EDFCloudPolicy())):
        results = backend.run_episode(
            tasks,
            AlwaysEdgePolicy(),
            EdgeForwardPolicy(),
            cloud_policy,
            seed=data["scenario"]["seed"],
        )
        ordered = sorted(results, key=lambda result: result.execution_start_time_s)
        print(label, [
            (
                result.task_id,
                round(result.execution_start_time_s, 6),
                round(result.end_to_end_latency_s, 6),
                result.deadline_met,
            )
            for result in ordered
        ])


if __name__ == "__main__":
    main()
