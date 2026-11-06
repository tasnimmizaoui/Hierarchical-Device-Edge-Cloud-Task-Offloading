"""Run a validated Alibaba task/instance subset through the SimPy backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from mec_offloading.baselines import (
    EDFCloudPolicy,
    EdgeExecutePolicy,
    GreedyNearestDevicePolicy,
)
from mec_offloading.simulation import SimPyBackend
from mec_offloading.trace_workloads import (
    TraceWorkloadConfig,
    load_alibaba_2017_batch_tasks,
    load_alibaba_batch_instances,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--format", choices=("2017-task", "2018-instance"), default="2017-task")
    args = parser.parse_args()
    loader = (
        load_alibaba_2017_batch_tasks
        if args.format == "2017-task"
        else load_alibaba_batch_instances
    )
    if args.format == "2017-task" and args.instances is None:
        parser.error("--instances is required for --format 2017-task")
    config = TraceWorkloadConfig(max_tasks=args.max_tasks)
    loaded = (
        loader(args.input, instance_path=args.instances, config=config)
        if args.format == "2017-task"
        else loader(args.input, config=config)
    )
    tasks = list(loaded.tasks)
    results = SimPyBackend().run_episode(
        tasks,
        GreedyNearestDevicePolicy(),
        EdgeExecutePolicy(),
        EDFCloudPolicy(),
        seed=7,
    )
    print(f"selected={len(tasks)}")
    print(f"task_rejected_before_limit={loaded.rejected_count}")
    print(f"instance_records_rejected={loaded.auxiliary_rejected_count}")
    print(f"stopped_early={loaded.stopped_early}")
    for result in results:
        print(result.task_id, result.route[-1].value, result.end_to_end_latency_s)


if __name__ == "__main__":
    main()
