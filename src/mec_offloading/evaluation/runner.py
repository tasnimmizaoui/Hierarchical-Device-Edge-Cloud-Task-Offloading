"""Reproducible baseline runs and CSV metric exports."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Protocol

from mec_offloading.domain.models import Task, TaskResult
from mec_offloading.domain.ports import CloudSchedulingPolicy, DevicePolicy, EdgePolicy
from mec_offloading.simulation import SimPyBackend


class PolicyBundle(Protocol):
    device: DevicePolicy
    edge: EdgePolicy
    cloud: CloudSchedulingPolicy


@dataclass(frozen=True, slots=True)
class BaselinePolicy:
    name: str
    device: DevicePolicy
    edge: EdgePolicy
    cloud: CloudSchedulingPolicy


PolicyBuilder = Callable[[int], Sequence[BaselinePolicy]]

METRIC_KEYS = (
    "mean_latency_s",
    "median_latency_s",
    "p95_latency_s",
    "deadline_success_rate",
    "deadline_miss_rate",
    "mean_device_energy_j",
    "throughput_tasks_per_s",
    "mean_communication_delay_s",
    "communication_bytes",
    "feedback_messages",
    "feedback_bytes",
    "mean_feedback_modeled_delay_s",
)


def percentile(values: list[float], percentage: float) -> float:
    """Linear-interpolated percentile without requiring NumPy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_results(
    policy_name: str,
    results: list[TaskResult],
    *,
    seed: int | None = None,
) -> dict[str, float | int | str]:
    latencies = [result.end_to_end_latency_s for result in results]
    completed = [result for result in results if result.failure_reason is None]
    completion_time = max((result.completion_time_s for result in results), default=0.0)
    row: dict[str, float | int | str] = {
        "policy": policy_name,
        "seed": "" if seed is None else seed,
        "task_count": len(results),
        "mean_latency_s": mean(latencies) if latencies else 0.0,
        "median_latency_s": median(latencies) if latencies else 0.0,
        "p95_latency_s": percentile(latencies, 95.0),
        "deadline_success_rate": (
            sum(result.deadline_met for result in results) / len(results) if results else 0.0
        ),
        "deadline_miss_rate": (
            sum(not result.deadline_met for result in results) / len(results) if results else 0.0
        ),
        "mean_device_energy_j": (
            mean(result.device_energy_j for result in completed) if completed else 0.0
        ),
        "throughput_tasks_per_s": len(completed) / completion_time if completion_time > 0 else 0.0,
        "mean_communication_delay_s": (
            mean(result.communication_delay_s for result in results) if results else 0.0
        ),
        "communication_bytes": sum(result.communication_bytes for result in results),
        "feedback_messages": sum(result.feedback_messages for result in results),
        "feedback_bytes": sum(result.feedback_bytes for result in results),
        "mean_feedback_modeled_delay_s": (
            mean(result.feedback_modeled_delay_s for result in results) if results else 0.0
        ),
    }
    return row


def summarize_runs(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Calculate across-seed mean and sample standard deviation per policy."""
    by_policy: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summaries: list[dict[str, float | int | str]] = []
    for policy, policy_rows in by_policy.items():
        summary: dict[str, float | int | str] = {
            "policy": policy,
            "seed_count": len(policy_rows),
        }
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in policy_rows]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def run_baselines(
    tasks_by_seed: dict[int, list[Task]],
    policies: Sequence[BaselinePolicy] | PolicyBuilder,
    *,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Run every policy on one identical workload per seed and export CSVs.

    Pass a callable policy builder when using multiple seeds. It receives the
    seed and must construct fresh policy instances, which resets random-policy
    state independently for every seed and policy.
    """
    if not tasks_by_seed:
        raise ValueError("at least one seed/workload is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = SimPyBackend()
    detailed_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    for seed, tasks in tasks_by_seed.items():
        seed_policies = list(policies(seed) if callable(policies) else policies)
        for policy in seed_policies:
            results = list(
                backend.run_episode(
                    tasks,
                    policy.device,
                    policy.edge,
                    policy.cloud,
                    seed=seed,
                )
            )
            aggregate_rows.append(aggregate_results(policy.name, results, seed=seed))
            detailed_rows.extend(
                {
                    "policy": policy.name,
                    "seed": seed,
                    "task_id": result.task_id,
                    "route": "->".join(tier.value for tier in result.route),
                    "arrival_time_s": result.arrival_time_s,
                    "completion_time_s": result.completion_time_s,
                    "execution_start_time_s": result.execution_start_time_s,
                    "end_to_end_latency_s": result.end_to_end_latency_s,
                    "deadline_met": result.deadline_met,
                    "device_energy_j": result.device_energy_j,
                    "communication_delay_s": result.communication_delay_s,
                    "communication_bytes": result.communication_bytes,
                    "feedback_messages": result.feedback_messages,
                    "feedback_bytes": result.feedback_bytes,
                    "feedback_modeled_delay_s": result.feedback_modeled_delay_s,
                    "failure_reason": result.failure_reason or "",
                }
                for result in results
            )

    detailed_path = output_dir / "task_results.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    statistics_path = output_dir / "aggregate_statistics.csv"
    _write_csv(detailed_path, detailed_rows)
    _write_csv(aggregate_path, aggregate_rows)
    _write_csv(statistics_path, summarize_runs(aggregate_rows))
    return detailed_path, aggregate_path, statistics_path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
