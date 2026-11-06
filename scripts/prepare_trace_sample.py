"""Validate a local Alibaba trace subset and report its audit counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from mec_offloading.trace_workloads import (
    load_alibaba_2017_batch_tasks,
    load_alibaba_batch_instances,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
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
    result = (
        loader(args.input, instance_path=args.instances)
        if args.format == "2017-task"
        else loader(args.input)
    )
    print(f"source={result.source}")
    print(f"selected={result.selected_count}")
    print(f"task_rejected={result.rejected_count}")
    print(f"instance_records_rejected={result.auxiliary_rejected_count}")
    print(f"stopped_early={result.stopped_early}")
    print("task_rejection_reasons=")
    task_reasons = sorted(
        result.rejection_reasons.items(), key=lambda item: (-item[1], item[0])
    )
    for reason, count in task_reasons:
        print(f"  {count}: {reason}")
    print("instance_rejection_reasons=")
    for reason, count in sorted(
        result.auxiliary_rejection_reasons.items(), key=lambda item: (-item[1], item[0])
    ):
        print(f"  {count}: {reason}")


if __name__ == "__main__":
    main()
