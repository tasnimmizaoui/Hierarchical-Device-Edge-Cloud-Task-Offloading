"""Run synthetic or Alibaba tasks through a configurable SimPy topology."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeForwardPolicy,
)
from mec_offloading.domain.models import DeviceState
from mec_offloading.evaluation.runner import aggregate_results
from mec_offloading.simulation import (
    EdgeConfig,
    SimPyBackend,
    SimulationConfig,
    TopologyConfig,
)
from mec_offloading.trace_workloads import (
    TraceWorkloadConfig,
    assign_tasks_to_devices,
    load_alibaba_2017_batch_tasks,
    load_alibaba_batch_instances,
)
from mec_offloading.workloads import generate_synthetic_tasks


def load_topology(path: Path) -> SimulationConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    topology_data = data["topology"]
    devices = tuple(
        DeviceState(
            device["id"],
            device["cpu_mips"],
            device["battery_j"],
            device["battery_capacity_j"],
            device["tx_power_w"],
            device["rx_power_w"],
            device["cpu_energy_per_mi_j"],
        )
        for device in topology_data["devices"]
    )
    edges = tuple(
        EdgeConfig(
            edge["id"],
            edge["cpu_mips"],
            edge["device_edge_bandwidth_mbps"],
            edge["edge_cloud_bandwidth_mbps"],
            edge["device_edge_propagation_s"],
            edge["edge_cloud_propagation_s"],
        )
        for edge in topology_data["edges"]
    )
    topology = TopologyConfig(
        devices,
        edges,
        tuple(topology_data["device_to_edge"].items()),
    )
    return SimulationConfig(
        cloud_cpu_capacity_mips=data["resources"]["cloud_cpu_mips"],
        topology=topology,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--source", choices=("synthetic", "alibaba2017", "alibaba2018"),
                        default="alibaba2017")
    parser.add_argument("--format", choices=("2017-task", "2018-instance"))
    parser.add_argument("--topology-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/trace-multi-topology"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    source = "alibaba2017" if args.format == "2017-task" else args.source
    source = "alibaba2018" if args.format == "2018-instance" else source
    if source != "synthetic" and args.input is None:
        parser.error("input is required for a trace source")
    if source == "alibaba2017" and args.instances is None:
        parser.error("--instances is required for --source alibaba2017")

    topology_config = args.topology_config or Path("configs/alibaba-multi-topology.yaml")
    simulation_config = load_topology(topology_config)
    device_ids = tuple(device.device_id for device in simulation_config.topology.devices)
    loaded = None
    if source == "synthetic":
        tasks = assign_tasks_to_devices(
            generate_synthetic_tasks(args.max_tasks, seed=args.seed),
            device_ids,
        )
    else:
        trace_config = TraceWorkloadConfig(
            max_tasks=args.max_tasks,
            device_ids=device_ids,
        )
        if source == "alibaba2017":
            loaded = load_alibaba_2017_batch_tasks(
                args.input, instance_path=args.instances, config=trace_config
            )
        else:
            loaded = load_alibaba_batch_instances(args.input, config=trace_config)
        tasks = list(loaded.tasks)

    results = SimPyBackend(config=simulation_config).run_episode(
        tasks,
        AlwaysEdgePolicy(),
        EdgeForwardPolicy(),
        EDFCloudPolicy(),
        seed=args.seed,
    )
    detailed_rows = [
        {
            "task_id": result.task_id,
            "source_device_id": result.source_device_id,
            "source_edge_id": result.source_edge_id or "",
            "route": "->".join(tier.value for tier in result.route),
            "arrival_time_s": result.arrival_time_s,
            "completion_time_s": result.completion_time_s,
            "end_to_end_latency_s": result.end_to_end_latency_s,
            "execution_start_time_s": result.execution_start_time_s,
            "device_energy_j": result.device_energy_j,
            "deadline_met": result.deadline_met,
        }
        for result in results
    ]
    aggregate = aggregate_results(f"{source}_multi_topology", results, seed=args.seed)
    write_csv(args.output_dir / "task_results.csv", detailed_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", [aggregate])
    print(f"source={source}")
    print(f"selected={len(tasks)}")
    if loaded is not None:
        print(f"task_rejected_before_limit={loaded.rejected_count}")
        print(f"instance_records_rejected={loaded.auxiliary_rejected_count}")
        print(f"stopped_early={loaded.stopped_early}")
        print(f"task_rejection_reasons={loaded.rejection_reasons}")
        print(f"instance_rejection_reasons={loaded.auxiliary_rejection_reasons}")
    print(f"topology_config={topology_config}")
    print(f"task_results={args.output_dir / 'task_results.csv'}")
    print(f"aggregate_metrics={args.output_dir / 'aggregate_metrics.csv'}")
    for row in detailed_rows:
        print(
            row["task_id"],
            row["source_device_id"],
            row["source_edge_id"],
            row["route"],
            f"latency={row['end_to_end_latency_s']:.6f}s",
            f"energy={row['device_energy_j']:.6f}J",
            f"deadline_met={row['deadline_met']}",
        )


if __name__ == "__main__":
    main()
