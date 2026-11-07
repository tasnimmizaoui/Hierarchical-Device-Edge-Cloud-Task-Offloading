"""Shared workload and topology inputs for integrated RL experiments."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

from mec_offloading.domain.models import Task
from mec_offloading.simulation.simpy_backend import SimulationConfig
from mec_offloading.simulation.topology_io import load_topology_config
from mec_offloading.trace_workloads import (
    ALIBABA_2017_ADAPTER_VERSION,
    TraceWorkloadConfig,
    load_alibaba_2017_batch_tasks,
    split_trace_tasks,
)

BEST_TIER_DEADLINE_MULTIPLIER = 0.9
BEST_TIER_DEADLINE_SLACK_S = 0.5
DEFAULT_TRACE_ARRIVAL_COMPRESSION = 0.1360670304522077


def compress_arrivals(tasks: tuple[Task, ...] | list[Task], factor: float) -> list[Task]:
    """Scale inter-arrival times around the first arrival; factor < 1 adds load."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("arrival compression factor must be finite and > 0")
    if not tasks:
        return []
    first_arrival = min(task.arrival_time_s for task in tasks)
    return [
        replace(
            task,
            arrival_time_s=first_arrival
            + (task.arrival_time_s - first_arrival) * factor,
        )
        for task in tasks
    ]


def offered_load(tasks: tuple[Task, ...] | list[Task], capacity_mips: float) -> float:
    """Compute aggregate MI offered per arrival-span relative to MIPS capacity."""
    if not math.isfinite(capacity_mips) or capacity_mips <= 0:
        raise ValueError("capacity_mips must be finite and > 0")
    if len(tasks) < 2:
        raise ValueError("at least two tasks are required to calculate offered load")
    span = max(task.arrival_time_s for task in tasks) - min(
        task.arrival_time_s for task in tasks
    )
    if span <= 0:
        raise ValueError("task arrival span must be > 0")
    return sum(task.required_mi for task in tasks) / (span * capacity_mips)


@dataclass(frozen=True)
class TraceExperimentData:
    source: str
    train: tuple[Task, ...]
    validation: tuple[Task, ...]
    test: tuple[Task, ...]
    simulation_config: SimulationConfig
    metadata: dict[str, object]


def load_trace_experiment(
    trace_path: str | Path,
    instances_path: str | Path,
    topology_path: str | Path,
    *,
    max_tasks: int,
    deadline_policy: str = "fixed-5s",
) -> TraceExperimentData:
    """Load disjoint chronological trace splits mapped onto a configured topology."""
    topology_file = Path(topology_path)
    simulation_config = load_topology_config(topology_file)
    topology = simulation_config.topology
    if topology is None:
        raise ValueError("Trace RL experiments require an explicit device/edge topology")
    device_ids = tuple(device.device_id for device in topology.devices)
    trace_config = TraceWorkloadConfig(
        max_tasks=max_tasks,
        device_ids=device_ids,
    )
    loaded = load_alibaba_2017_batch_tasks(
        trace_path,
        instance_path=instances_path,
        config=trace_config,
    )
    splits = split_trace_tasks(loaded.tasks)
    if deadline_policy == "local-relative-2x":
        device_capacity = {
            device.device_id: device.cpu_capacity_mips for device in topology.devices
        }
        splits = {
            split_name: tuple(
                replace(
                    task,
                    deadline_s=2.0 * task.required_mi / device_capacity[task.device_id],
                )
                for task in split_tasks
            )
            for split_name, split_tasks in splits.items()
        }
    elif deadline_policy == "best-tier-calibrated":
        best_capacity_mips = max(
            [device.cpu_capacity_mips for device in topology.devices]
            + [edge.cpu_capacity_mips for edge in topology.edges]
            + [simulation_config.cloud_cpu_capacity_mips]
        )
        splits = {
            split_name: tuple(
                replace(
                    task,
                    deadline_s=max(
                        1.0,
                        BEST_TIER_DEADLINE_MULTIPLIER
                        * task.required_mi
                        / best_capacity_mips
                        + BEST_TIER_DEADLINE_SLACK_S,
                    ),
                )
                for task in split_tasks
            )
            for split_name, split_tasks in splits.items()
        }
    elif deadline_policy != "fixed-5s":
        raise ValueError(f"Unsupported trace deadline policy: {deadline_policy}")
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError(
            "The selected trace subset must contain at least one task in each "
            "train/validation/test split"
        )
    metadata: dict[str, object] = {
        "source": "alibaba2017",
        "trace_adapter_version": ALIBABA_2017_ADAPTER_VERSION,
        "trace_path": str(trace_path),
        "instances_path": str(instances_path),
        "trace_sha256": hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),
        "instances_sha256": hashlib.sha256(Path(instances_path).read_bytes()).hexdigest(),
        "selected_tasks": loaded.selected_count,
        "rejected_tasks": loaded.rejected_count,
        "rejected_instance_records": loaded.auxiliary_rejected_count,
        "stopped_early": loaded.stopped_early,
        "train_tasks": len(splits["train"]),
        "validation_tasks": len(splits["validation"]),
        "test_tasks": len(splits["test"]),
        "topology_config": str(topology_file),
        "topology_sha256": hashlib.sha256(topology_file.read_bytes()).hexdigest(),
        "devices": [device.device_id for device in topology.devices],
        "edges": [edge.edge_id for edge in topology.edges],
        "device_to_edge": dict(topology.device_to_edge),
        "cloud_cpu_capacity_mips": simulation_config.cloud_cpu_capacity_mips,
        "resource_concurrency": "one execution slot per device/edge and one cloud worker",
        "task_assumptions": {
            "reference_mips_per_core": trace_config.reference_mips_per_core,
            "input_size_mb": trace_config.input_size_mb,
            "input_size_distribution": (
                {"type": "lognormal", "median_mb": 1.0, "sigma": 1.0, "cap_mb": 50.0}
                if trace_config.input_size_mb is None
                else {"type": "fixed", "size_mb": trace_config.input_size_mb}
            ),
            "result_size_mb": trace_config.result_size_mb,
            "deadline_s": (
                trace_config.deadline_s if deadline_policy == "fixed-5s" else None
            ),
            "relative_deadline_multiplier": (
                2.0 if deadline_policy == "local-relative-2x" else None
            ),
            "best_tier_deadline_multiplier": (
                BEST_TIER_DEADLINE_MULTIPLIER
                if deadline_policy == "best-tier-calibrated"
                else None
            ),
            "best_tier_deadline_slack_s": (
                BEST_TIER_DEADLINE_SLACK_S
                if deadline_policy == "best-tier-calibrated"
                else None
            ),
            "device_assignment": "round-robin across configured devices",
            "deadline_policy": deadline_policy,
            "relative_deadline_definition": (
                "2x isolated local execution time"
                if deadline_policy == "local-relative-2x"
                else (
                    "0.9x work at fastest configured MIPS capacity plus 0.5 seconds, "
                    "with a 1-second minimum; synthetic assumption"
                    if deadline_policy == "best-tier-calibrated"
                    else "fixed 5 seconds; synthetic assumption"
                )
            ),
        },
    }
    return TraceExperimentData(
        source="alibaba2017",
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        simulation_config=simulation_config,
        metadata=metadata,
    )


def sample_tasks(tasks: tuple[Task, ...], count: int, seed: int) -> list[Task]:
    """Select a reproducible subset without replacement, or return the whole split."""
    if count <= 0:
        raise ValueError("count must be > 0")
    if count >= len(tasks):
        return list(tasks)
    return random.Random(seed).sample(tasks, count)


def topology_metadata(config: SimulationConfig) -> dict[str, object]:
    """Return a compact topology signature for checkpoint compatibility checks."""
    topology = config.topology
    if topology is None:
        return {
            "devices": ["device-1"],
            "edges": ["edge-1"],
            "device_to_edge": {"device-1": "edge-1"},
            "cloud_cpu_capacity_mips": config.cloud_cpu_capacity_mips,
        }
    return {
        "devices": [device.device_id for device in topology.devices],
        "edges": [edge.edge_id for edge in topology.edges],
        "device_to_edge": dict(topology.device_to_edge),
        "cloud_cpu_capacity_mips": config.cloud_cpu_capacity_mips,
    }
