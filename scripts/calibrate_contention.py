"""Sweep trace-arrival compression and report load, success, and queue waits."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    DelayedGreedyComputeEdgePolicy,
    DelayedGreedyNearestDevicePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    GreedyComputeEdgePolicy,
    GreedyNearestDevicePolicy,
)
from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.rl.dqn import DQNAgent
from mec_offloading.rl.experiment_data import (
    compress_arrivals,
    load_trace_experiment,
    offered_load,
)
from mec_offloading.rl.integrated import run_centralized, run_hierarchical
from mec_offloading.simulation import SimPyBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=Path, default=Path("data/raw/batch_task.csv"))
    parser.add_argument(
        "--trace-instances", type=Path, default=Path("data/raw/batch_instance.csv")
    )
    parser.add_argument(
        "--topology-config", type=Path,
        default=Path("configs/alibaba-multi-topology.yaml"),
    )
    parser.add_argument("--max-trace-tasks", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feedback", choices=["none", "periodic", "frequent", "event"],
                        default="periodic")
    parser.add_argument("--feedback-frequency", type=int, default=5)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/contention-calibration"))
    parser.add_argument("--factors", type=float, nargs="+", default=[1.0, .5, .2, .1, .05])
    parser.add_argument("--target-load", type=float, default=.75)
    args = parser.parse_args()
    if args.max_trace_tasks < 3 or args.feedback_frequency <= 0:
        parser.error("max trace tasks must be >= 3 and feedback frequency must be > 0")
    if (
        not args.factors
        or any(factor <= 0 for factor in args.factors)
        or args.target_load <= 0
    ):
        parser.error("arrival factors and target load must be > 0")

    data = load_trace_experiment(
        args.trace_path,
        args.trace_instances,
        args.topology_config,
        max_tasks=args.max_trace_tasks,
        deadline_policy="best-tier-calibrated",
    )
    tasks = list(data.test)
    selected = offered_load(tasks, 12_500.0) / args.target_load
    factors = sorted(set(args.factors) | {selected})
    agents = None
    freeze_cloud = True
    if args.models is not None:
        agents = {
            "device": DQNAgent.load(args.models / "device.pt"),
            "edge": DQNAgent.load(args.models / "edge.pt"),
            "centralized": DQNAgent.load(args.models / "centralized.pt"),
        }
        cloud_path = args.models / "cloud.pt"
        if cloud_path.is_file():
            agents["cloud"] = DQNAgent.load(cloud_path)
            freeze_cloud = False

    rows = []
    for factor in factors:
        compressed = compress_arrivals(tasks, factor)
        backend = SimPyBackend(config=data.simulation_config)
        load = offered_load(compressed, 12_500.0)
        policy_runs = [
            (
                "always_edge",
                backend.run_episode(
                    compressed, AlwaysEdgePolicy(), EdgeExecutePolicy(),
                    EDFCloudPolicy(), seed=args.seed,
                ),
            ),
            (
                "greedy_oracle_edf",
                backend.run_episode(
                    compressed, GreedyNearestDevicePolicy(), GreedyComputeEdgePolicy(),
                    EDFCloudPolicy(), seed=args.seed,
                ),
            ),
        ]
        coordinator = FeedbackCoordinator(args.feedback, args.feedback_frequency)
        policy_runs.append((
            "greedy_delayed",
            backend.run_episode(
                compressed,
                DelayedGreedyNearestDevicePolicy(coordinator),
                DelayedGreedyComputeEdgePolicy(coordinator),
                EDFCloudPolicy(),
                seed=args.seed,
            ),
        ))
        if agents is not None:
            policy_runs.append((
                f"hierarchical_{args.feedback}",
                run_hierarchical(
                    compressed, agents, seed=args.seed, backend=backend,
                    feedback_mode=args.feedback,
                    feedback_frequency=args.feedback_frequency,
                    freeze_cloud=freeze_cloud,
                )[0],
            ))
            policy_runs.append((
                "centralized",
                run_centralized(
                    compressed, agents["centralized"], seed=args.seed, backend=backend,
                )[0],
            ))
        for policy, results in policy_runs:
            rows.append({
                "arrival_compression": factor,
                "aggregate_edge_offered_load": load,
                "policy": policy,
                "raw_success_rate": (
                    sum(result.deadline_met for result in results) / len(results)
                    if results else 0.0
                ),
                "queue_wait_over_1s_share": (
                    sum(result.queue_wait_s > 1 for result in results) / len(results)
                    if results else 0.0
                ),
                "mean_device_energy_j": (
                    sum(result.device_energy_j for result in results) / len(results)
                    if results else 0.0
                ),
                "communication_bytes": sum(
                    result.communication_bytes for result in results
                ),
                "feedback_bytes": sum(result.feedback_bytes for result in results),
                "total_bytes": sum(
                    result.communication_bytes + result.feedback_bytes
                    for result in results
                ),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "contention_calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("k, edge load, policy, raw success, queue wait >1s")
    for row in rows:
        print(
            f"{row['arrival_compression']:.3g}, "
            f"{row['aggregate_edge_offered_load']:.3f}, {row['policy']}, "
            f"{row['raw_success_rate']:.3f}, {row['queue_wait_over_1s_share']:.3f}"
        )
    print(
        f"Selected compression factor for target load {args.target_load:g}: "
        f"{selected:.6f}"
    )


if __name__ == "__main__":
    main()
