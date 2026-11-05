"""Run seeded baseline comparisons and export CSV metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    FIFOCloudPolicy,
    GreedyNearestDevicePolicy,
    LocalOnlyPolicy,
    RandomCloudPolicy,
    RandomDevicePolicy,
    RandomEdgePolicy,
)
from mec_offloading.evaluation import BaselinePolicy, run_baselines
from mec_offloading.workloads import generate_synthetic_tasks


def build_policies(seed: int) -> list[BaselinePolicy]:
    """Construct fresh policies; random streams are independent per seed/run."""
    return [
        BaselinePolicy("local", LocalOnlyPolicy(), EdgeExecutePolicy(), EDFCloudPolicy()),
        BaselinePolicy(
            "greedy_edge",
            GreedyNearestDevicePolicy(),
            EdgeExecutePolicy(),
            EDFCloudPolicy(),
        ),
        BaselinePolicy("edge_execute", AlwaysEdgePolicy(), EdgeExecutePolicy(), EDFCloudPolicy()),
        BaselinePolicy("cloud_edf", AlwaysEdgePolicy(), EdgeForwardPolicy(), EDFCloudPolicy()),
        BaselinePolicy("cloud_fifo", AlwaysEdgePolicy(), EdgeForwardPolicy(), FIFOCloudPolicy()),
        BaselinePolicy(
            "random",
            RandomDevicePolicy(seed + 101),
            RandomEdgePolicy(seed + 202),
            RandomCloudPolicy(seed + 303),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dev-smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_by_seed = {
        seed: generate_synthetic_tasks(count=args.task_count, seed=seed)
        for seed in args.seeds
    }
    detailed, aggregate, statistics = run_baselines(
        tasks_by_seed,
        build_policies,
        output_dir=args.output_dir,
    )
    print(f"Wrote {detailed}")
    print(f"Wrote {aggregate}")
    print(f"Wrote {statistics}")


if __name__ == "__main__":
    main()
