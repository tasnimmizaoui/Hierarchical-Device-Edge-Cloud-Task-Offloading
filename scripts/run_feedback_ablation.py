"""Train and evaluate reproducible multi-seed Alibaba feedback ablations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--feedback-modes", nargs="+",
                        choices=["none", "periodic", "frequent", "event"],
                        default=["none", "periodic", "frequent", "event"])
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+",
                        default=[10_001, 10_002, 10_003, 10_004, 10_005])
    parser.add_argument("--event-thresholds", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5])
    parser.add_argument("--feedback-frequency", type=int, default=5)
    parser.add_argument("--max-trace-tasks", type=int, default=5_000)
    parser.add_argument("--arrival-compression", type=float)
    parser.add_argument("--learn-cloud", action="store_true",
                        help="Train the third tier instead of using fixed cloud EDF")
    parser.add_argument("--models-root", type=Path,
                        default=Path("artifacts/models/feedback-ablation"))
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/feedback-ablation"))
    parser.add_argument("--credit-mode", choices=["terminal", "next-decision"],
                        default="terminal")
    parser.add_argument("--algorithms", choices=["dqn", "double-dqn"], nargs="+",
                        default=["double-dqn"])
    args = parser.parse_args()
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or not args.feedback_modes
        or len(set(args.feedback_modes)) != len(args.feedback_modes)
        or args.episodes <= 0
        or args.tasks <= 0
        or args.feedback_frequency <= 0
        or args.max_trace_tasks < 3
        or not args.evaluation_seeds
        or not args.algorithms
        or not args.event_thresholds
        or len(set(args.event_thresholds)) != len(args.event_thresholds)
        or any(not 0 <= threshold <= 1 for threshold in args.event_thresholds)
    ):
        parser.error("Seeds/modes must be unique and all counts must be positive")

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_integrated_rl.py"
    compare_script = script_dir / "compare_integrated_policies.py"
    for mode in args.feedback_modes:
        thresholds = args.event_thresholds if mode == "event" else [0.25]
        for threshold in thresholds:
            for algorithm in args.algorithms:
                for seed in args.seeds:
                    condition = (
                        f"fb-{mode}-threshold-{threshold:g}-{algorithm}-seed-{seed}-"
                        f"{args.credit_mode}"
                    )
                    model_dir = args.models_root / condition
                    result_dir = args.results_root / condition
                    compression_args = (
                        ["--arrival-compression", str(args.arrival_compression)]
                        if args.arrival_compression is not None else []
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(train_script),
                            "--source", "alibaba2017",
                            "--deadline-policy", "best-tier-calibrated",
                            "--topology-augmentation", "sweep",
                            "--feedback", mode,
                            "--event-threshold", str(threshold),
                            "--feedback-frequency", str(args.feedback_frequency),
                            "--algorithm", algorithm,
                            "--credit-mode", args.credit_mode,
                            "--seed", str(seed),
                            "--episodes", str(args.episodes),
                            "--tasks", str(args.tasks),
                            "--max-trace-tasks", str(args.max_trace_tasks),
                            "--output", str(model_dir),
                            *compression_args,
                            *(["--learn-cloud"] if args.learn_cloud else []),
                        ],
                        check=True,
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(compare_script),
                            "--source", "alibaba2017",
                            "--deadline-policy", "best-tier-calibrated",
                            "--feedback", mode,
                            "--feedback-modes", mode,
                            "--event-thresholds", *map(str, args.event_thresholds),
                            "--feedback-frequency", str(args.feedback_frequency),
                            "--seeds", *map(str, args.evaluation_seeds),
                            "--tasks", str(args.max_trace_tasks),
                            "--max-trace-tasks", str(args.max_trace_tasks),
                            "--models", str(model_dir),
                            "--output", str(result_dir),
                            *compression_args,
                        ],
                        check=True,
                    )


if __name__ == "__main__":
    main()
