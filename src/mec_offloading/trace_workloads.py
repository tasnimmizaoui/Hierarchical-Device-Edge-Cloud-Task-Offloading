"""Alibaba Cluster Trace v2018 workload adapter.

The adapter consumes a small, locally obtained ``batch_instance.csv`` subset.
Alibaba provides datacenter execution records, not complete MEC task records,
so missing MEC fields are explicit configuration assumptions.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist

from mec_offloading.domain.models import Task

_DEPENDENCY_SUFFIX = re.compile(r"(?:^|_)M?\d+(?:_\d+)*$")
_DEFAULT_COMPLETED_STATUSES = frozenset(
    {"completed", "finished", "success", "succeeded", "terminated"}
)
_ALIBABA_2017_BATCH_TASK_FIELDS = (
    "create_timestamp",
    "modify_timestamp",
    "job_id",
    "task_id",
    "instance_num",
    "status",
    "plan_cpu",
    "plan_mem",
)
ALIBABA_2017_ADAPTER_VERSION = (
    "alibaba2017-mean-completed-instance-lognormal-payload-v2"
)


@dataclass(frozen=True, slots=True)
class TraceWorkloadConfig:
    """Explicit assumptions used to complete the shared :class:`Task` contract."""

    reference_mips_per_core: float = 1_000.0
    input_size_mb: float | None = None
    result_size_mb: float = 0.1
    deadline_s: float = 5.0
    device_ids: tuple[str, ...] = ("device-1",)
    accepted_statuses: frozenset[str] = _DEFAULT_COMPLETED_STATUSES
    max_tasks: int | None = None
    reject_dag_tasks: bool = True

    def __post_init__(self) -> None:
        if self.reference_mips_per_core <= 0:
            raise ValueError("reference_mips_per_core must be > 0")
        if (
            self.input_size_mb is not None and self.input_size_mb < 0
        ) or self.result_size_mb < 0:
            raise ValueError("payload sizes must be >= 0")
        if self.deadline_s <= 0:
            raise ValueError("deadline_s must be > 0")
        if not self.device_ids or any(not device_id for device_id in self.device_ids):
            raise ValueError("device_ids must contain at least one non-empty ID")
        if self.max_tasks is not None and self.max_tasks <= 0:
            raise ValueError("max_tasks must be > 0 when provided")


def payload_mb(
    task_id: str, median: float = 1.0, sigma: float = 1.0, cap: float = 50.0
) -> float:
    """Return a stable, task-specific lognormal payload size in decimal MB."""
    if (
        not task_id
        or not all(math.isfinite(value) for value in (median, sigma, cap))
        or median <= 0
        or sigma < 0
        or cap <= 0
    ):
        raise ValueError("task_id, median, and cap must be positive; sigma must be >= 0")
    uniform = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    quantile = min(max(uniform, 1e-6), 1.0 - 1e-6)
    normal = NormalDist().inv_cdf(quantile)
    return min(cap, median * math.exp(sigma * normal))


def _input_size_mb(task_id: str, config: TraceWorkloadConfig) -> float:
    return (
        config.input_size_mb
        if config.input_size_mb is not None
        else payload_mb(task_id)
    )


@dataclass(frozen=True, slots=True)
class TraceReject:
    """One explicitly rejected source record."""

    row_number: int
    reason: str


@dataclass(frozen=True, slots=True)
class TraceLoadResult:
    """Tasks plus audit information for one deterministic preprocessing run."""

    tasks: tuple[Task, ...]
    rejected: tuple[TraceReject, ...]
    source: str
    auxiliary_rejected: tuple[TraceReject, ...] = ()
    stopped_early: bool = False

    @property
    def selected_count(self) -> int:
        return len(self.tasks)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def auxiliary_rejected_count(self) -> int:
        return len(self.auxiliary_rejected)

    @property
    def rejection_reasons(self) -> dict[str, int]:
        return dict(Counter(rejection.reason for rejection in self.rejected))

    @property
    def auxiliary_rejection_reasons(self) -> dict[str, int]:
        return dict(Counter(rejection.reason for rejection in self.auxiliary_rejected))


def assign_tasks_to_devices(
    tasks: Iterable[Task], device_ids: tuple[str, ...]
) -> list[Task]:
    """Assign tasks round-robin to configured device IDs."""
    if not device_ids or any(not device_id for device_id in device_ids):
        raise ValueError("device_ids must contain at least one non-empty ID")
    return [
        replace(task, device_id=device_ids[index % len(device_ids)])
        for index, task in enumerate(tasks)
    ]


def _required_float(row: Mapping[str, str], field: str) -> float:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"missing {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"invalid {field}")
    return number


def _is_dag_task(task_name: str) -> bool:
    parts = task_name.split("_")
    return len(parts) > 1 and all(part.isdigit() for part in parts[1:])


def _task_from_instance(
    row: Mapping[str, str],
    *,
    trace_start_s: float,
    config: TraceWorkloadConfig,
    index: int,
) -> Task:
    instance_name = row.get("instance_name", "").strip()
    task_name = row.get("task_name", "").strip()
    job_name = row.get("job_name", "").strip()
    if not instance_name or not task_name or not job_name:
        raise ValueError("missing instance_name, task_name, or job_name")
    status = row.get("status", "").strip().casefold()
    if status not in config.accepted_statuses:
        raise ValueError(f"unaccepted status: {status or '<missing>'}")
    if config.reject_dag_tasks and _is_dag_task(task_name):
        raise ValueError("DAG-dependent task")

    start_time = _required_float(row, "start_time")
    end_time = _required_float(row, "end_time")
    cpu_avg = _required_float(row, "cpu_avg")
    if start_time < trace_start_s or end_time <= start_time:
        raise ValueError("invalid timestamps")
    if cpu_avg <= 0 or cpu_avg > 100:
        raise ValueError("cpu_avg must be in (0, 100]")

    duration_s = end_time - start_time
    required_mi = duration_s * (cpu_avg / 100.0) * config.reference_mips_per_core
    task_id = f"{job_name}/{task_name}/{instance_name}"
    return Task(
        task_id=task_id,
        device_id=config.device_ids[index % len(config.device_ids)],
        arrival_time_s=start_time - trace_start_s,
        input_size_mb=_input_size_mb(task_id, config),
        required_mi=required_mi,
        result_size_mb=config.result_size_mb,
        deadline_s=config.deadline_s,
    )


def load_alibaba_batch_instances(
    path: str | Path,
    *,
    config: TraceWorkloadConfig | None = None,
    trace_start_s: float | None = None,
) -> TraceLoadResult:
    """Load a deterministic, validated subset of Alibaba batch instances.

    ``batch_instance.csv`` does not contain task creation time. Its observed
    ``start_time`` is therefore used as the workload arrival proxy and is
    recorded relative to the minimum accepted source timestamp unless an
    explicit origin is supplied.
    """
    config = config or TraceWorkloadConfig()
    source = str(path)
    rows: list[tuple[int, dict[str, str]]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            rows.append((row_number, row))

    if trace_start_s is None:
        starts = []
        for _, row in rows:
            try:
                starts.append(_required_float(row, "start_time"))
            except ValueError:
                continue
        if not starts:
            raise ValueError("no valid start_time values found")
        trace_start_s = min(starts)

    tasks: list[Task] = []
    rejected: list[TraceReject] = []
    stopped_early = False
    seen_ids: set[str] = set()
    for row_number, row in rows:
        if config.max_tasks is not None and len(tasks) >= config.max_tasks:
            stopped_early = True
            break
        try:
            task = _task_from_instance(
                row, trace_start_s=trace_start_s, config=config, index=len(tasks)
            )
            if task.task_id in seen_ids:
                raise ValueError("duplicate task identifier")
        except (TypeError, ValueError) as error:
            rejected.append(TraceReject(row_number, str(error)))
            continue
        seen_ids.add(task.task_id)
        tasks.append(task)

    return TraceLoadResult(tuple(tasks), tuple(rejected), source, stopped_early=stopped_early)


def load_alibaba_2017_batch_tasks(
    path: str | Path,
    *,
    instance_path: str | Path | None = None,
    config: TraceWorkloadConfig | None = None,
    trace_start_s: float = 0.0,
) -> TraceLoadResult:
    """Load Alibaba 2017's headerless ``batch_task.csv``.

    The 2017 task table supplies creation time and requested CPU. The matching
    instance table supplies successful execution duration and actual CPU usage,
    which are used to derive MI. Only jobs represented by one task row are
    accepted; this avoids silently treating an unknown DAG as independent work.
    """
    config = config or TraceWorkloadConfig()
    if instance_path is None:
        raise ValueError("instance_path is required for the 2017 adapter")
    source = str(path)
    rows: list[tuple[int, dict[str, str]]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row_number, values in enumerate(csv.reader(handle), start=1):
            if not values or not any(value.strip() for value in values):
                continue
            if len(values) != len(_ALIBABA_2017_BATCH_TASK_FIELDS):
                rows.append((row_number, {}))
                continue
            rows.append(
                (row_number, dict(zip(_ALIBABA_2017_BATCH_TASK_FIELDS, values, strict=True)))
            )

    jobs: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for row_number, row in rows:
        job_id = row.get("job_id", "").strip()
        jobs.setdefault(job_id, []).append((row_number, row))

    tasks: list[Task] = []
    rejected: list[TraceReject] = []
    auxiliary_rejected: list[TraceReject] = []
    seen_ids: set[str] = set()
    origin = trace_start_s

    instance_usage: dict[tuple[str, str], tuple[float, int]] = {}
    with Path(instance_path).open(newline="", encoding="utf-8") as handle:
        for row_number, values in enumerate(csv.reader(handle), start=1):
            if not values or not any(value.strip() for value in values):
                continue
            if len(values) != 12:
                auxiliary_rejected.append(
                    TraceReject(row_number, "invalid batch_instance column count")
                )
                continue
            (
                start_timestamp,
                end_timestamp,
                instance_job_id,
                instance_task_id,
                _machine_id,
                status,
                _seq_no,
                _total_seq_no,
                _real_cpu_max,
                real_cpu_avg,
                _real_mem_max,
                _real_mem_avg,
            ) = values
            if status.strip().casefold() not in config.accepted_statuses:
                continue
            try:
                start = float(start_timestamp)
                end = float(end_timestamp)
                cpu = float(real_cpu_avg)
                if not all(math.isfinite(value) for value in (start, end, cpu)):
                    raise ValueError
                if end <= start or cpu <= 0:
                    raise ValueError
            except ValueError:
                auxiliary_rejected.append(
                    TraceReject(row_number, "invalid completed instance metrics")
                )
                continue
            key = (instance_job_id.strip(), instance_task_id.strip())
            work_mi, instance_count = instance_usage.get(key, (0.0, 0))
            instance_usage[key] = (
                work_mi + (end - start) * cpu * config.reference_mips_per_core,
                instance_count + 1,
            )

    stopped_early = False
    for row_number, row in rows:
        if config.max_tasks is not None and len(tasks) >= config.max_tasks:
            stopped_early = True
            break
        try:
            job_id = row["job_id"].strip()
            if not job_id or len(jobs.get(job_id, [])) != 1:
                raise ValueError("job has multiple task rows; DAG independence unknown")
            status = row["status"].strip().casefold()
            if status not in config.accepted_statuses:
                raise ValueError(f"unaccepted status: {status or '<missing>'}")
            create_time = _required_float(row, "create_timestamp")
            plan_cpu = _required_float(row, "plan_cpu")
            if create_time < origin or plan_cpu <= 0:
                raise ValueError("invalid timestamp or plan_cpu")
            task_id = f"job-{job_id}/task-{row['task_id'].strip()}"
            if not row["task_id"].strip():
                raise ValueError("missing task_id")
            if task_id in seen_ids:
                raise ValueError("duplicate task identifier")
            key = (job_id, row["task_id"].strip())
            total_instance_mi, instance_count = instance_usage.get(key, (0.0, 0))
            if instance_count == 0 or total_instance_mi <= 0:
                raise ValueError("no valid completed instances")
            required_mi = total_instance_mi / instance_count
            task = Task(
                task_id=task_id,
                device_id=config.device_ids[len(tasks) % len(config.device_ids)],
                arrival_time_s=create_time - origin,
                input_size_mb=_input_size_mb(task_id, config),
                required_mi=required_mi,
                result_size_mb=config.result_size_mb,
                deadline_s=config.deadline_s,
            )
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(TraceReject(row_number, str(error)))
            continue
        seen_ids.add(task.task_id)
        tasks.append(task)

    return TraceLoadResult(
        tuple(tasks),
        tuple(rejected),
        source,
        auxiliary_rejected=tuple(auxiliary_rejected),
        stopped_early=stopped_early,
    )


def split_trace_tasks(
    tasks: Iterable[Task],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, tuple[Task, ...]]:
    """Split chronologically into disjoint train, validation, and test windows."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    ordered = sorted(tasks, key=lambda task: (task.arrival_time_s, task.task_id))
    train_end = int(len(ordered) * train_fraction)
    validation_end = train_end + int(len(ordered) * validation_fraction)
    return {
        "train": tuple(ordered[:train_end]),
        "validation": tuple(ordered[train_end:validation_end]),
        "test": tuple(ordered[validation_end:]),
    }


def load_workload(
    source: str,
    *,
    count: int = 6,
    seed: int = 7,
    trace_path: str | Path | None = None,
    trace_instance_path: str | Path | None = None,
    trace_config: TraceWorkloadConfig | None = None,
) -> list[Task]:
    """Select either the preserved synthetic generator or the trace adapter."""
    if source == "synthetic":
        from mec_offloading.workloads import generate_synthetic_tasks

        return generate_synthetic_tasks(count=count, seed=seed)
    if source == "alibaba":
        if trace_path is None:
            raise ValueError("trace_path is required for source='alibaba'")
        return list(load_alibaba_batch_instances(trace_path, config=trace_config).tasks)
    if source == "alibaba2017":
        if trace_path is None:
            raise ValueError("trace_path is required for source='alibaba2017'")
        return list(
            load_alibaba_2017_batch_tasks(
                trace_path, instance_path=trace_instance_path, config=trace_config
            ).tasks
        )
    raise ValueError(f"unknown workload source: {source}")
