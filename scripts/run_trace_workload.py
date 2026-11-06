"""Run a validated Alibaba batch-instance subset through the SimPy backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from mec_offloading.baselines import (
    EDFCloudPolicy,
    EdgeExecutePolicy,
    GreedyNearestDevicePolicy,
)
from mec_offloading.simulation import SimPyBackend
from mec_offloading.trace_workloads import load_alibaba_batch_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-tasks", type=int, default=20)
    args = parser.parse_args()
    loaded = load_alibaba_batch_instances(
        args.input,
        config=None,
    )
    tasks = list(loaded.tasks[: args.max_tasks])
    results = SimPyBackend().run_episode(
        tasks,
        GreedyNearestDevicePolicy(),
        EdgeExecutePolicy(),
        EDFCloudPolicy(),
        seed=7,
    )
    print(f"selected={len(tasks)} rejected={loaded.rejected_count}")
    for result in results:
        print(result.task_id, result.route[-1].value, result.end_to_end_latency_s)


if __name__ == "__main__":
    main()
