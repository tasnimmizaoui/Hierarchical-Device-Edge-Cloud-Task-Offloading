"""Summarize success, device energy, and communication bytes across ablation runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from statistics import fmean, stdev

METRICS = (
    "deadline_success_rate",
    "p95_latency_s",
    "mean_device_energy_j",
    "communication_bytes",
    "feedback_bytes",
    "total_bytes",
    "mean_feedback_age_s",
)


def verify_protocol(paths: list[Path]) -> None:
    signatures = set()
    configuration_hashes: dict[tuple[object, ...], set[str]] = {}
    for path in paths:
        metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
        workload = metadata["workload"]
        signatures.add((
            metadata["arrival_compression"],
            metadata["deadline_policy"],
            metadata["cloud_scheduler"],
            metadata["feedback_frequency"],
            tuple(metadata["test_seeds"]),
            tuple(metadata["scenarios"]),
            workload["trace_adapter_version"],
            workload["trace_sha256"],
            workload["instances_sha256"],
            workload["topology_sha256"],
        ))
        condition = (
            metadata.get("feedback"),
            metadata.get("training_event_threshold"),
            metadata.get("training_algorithm"),
        )
        configuration_hashes.setdefault(condition, set()).add(
            str(metadata.get("configuration_hash", ""))
        )
    if len(signatures) > 1:
        raise ValueError(
            "Mixed evaluation protocols. Match compression, scheduler, seeds, "
            "scenarios, trace and topology before pooling results."
        )
    if any(len(hashes) != 1 or "" in hashes for hashes in configuration_hashes.values()):
        raise ValueError(
            "Configuration hashes differ across training seeds within an ablation."
        )


def _bootstrap_interval(values: list[float], *, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        fmean(rng.choices(values, k=len(values)))
        for _ in range(2_000)
    )
    return means[49], means[1_949]


def summarize_files(
    files: list[tuple[str, float, str, int, Path]], scenario: str
) -> list[dict[str, float | int | str]]:
    grouped: dict[
        tuple[str, float, str, str, str], dict[str, dict[str, list[float]]]
    ] = {}
    hashes: dict[tuple[str, float, str], str] = {}
    for mode, training_threshold, algorithm, training_seed, path in files:
        metadata = json.loads(
            path.with_name("metadata.json").read_text(encoding="utf-8")
        )
        hashes[(mode, training_threshold, algorithm)] = str(
            metadata["configuration_hash"]
        )
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not set(("scenario", "policy", *METRICS)).issubset(
                reader.fieldnames or ()
            ):
                raise ValueError(f"{path} is missing required metric columns")
            for row in reader:
                if row["scenario"] != scenario:
                    continue
                evaluation_threshold = row.get("event_threshold", "")
                group_key = (
                    mode,
                    training_threshold,
                    evaluation_threshold,
                    algorithm,
                    row["policy"],
                )
                values = grouped.setdefault(
                    group_key,
                    {},
                )
                for metric in METRICS:
                    per_seed = values.setdefault(metric, {})
                    per_seed.setdefault(str(training_seed), []).append(float(row[metric]))

    summaries = []
    for (mode, train_threshold, eval_threshold, algorithm, policy), metrics in sorted(
        grouped.items()
    ):
        deadline_rows = metrics["deadline_success_rate"]
        seed_count = len(deadline_rows)
        row: dict[str, float | int | str] = {
            "feedback_mode": mode,
            "training_event_threshold": train_threshold,
            "evaluation_event_threshold": eval_threshold,
            "algorithm": algorithm,
            "policy": policy,
            "training_seed_count": seed_count,
            "evaluation_rows": sum(len(values) for values in deadline_rows.values()),
            "configuration_hash": hashes[(mode, train_threshold, algorithm)],
        }
        for metric, per_seed in metrics.items():
            seed_means = [fmean(values) for _, values in sorted(per_seed.items())]
            row[f"{metric}_mean"] = fmean(seed_means)
            row[f"{metric}_sd"] = stdev(seed_means) if len(seed_means) > 1 else 0.0
            lower, upper = _bootstrap_interval(
                seed_means,
                seed=int.from_bytes(hashlib.sha256(
                    json.dumps(
                        [mode, train_threshold, eval_threshold, algorithm, metric]
                    ).encode("utf-8")
                ).digest()[:4], "big"),
            )
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        summaries.append(row)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/feedback-ablation"))
    parser.add_argument("--feedback-modes", nargs="+",
                        choices=["none", "periodic", "frequent", "event"],
                        default=["none", "periodic", "frequent", "event"])
    parser.add_argument("--event-thresholds", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5])
    parser.add_argument("--algorithms", choices=["dqn", "double-dqn"], nargs="+",
                        default=["double-dqn"])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--credit-mode", choices=["terminal", "next-decision"],
                        default="terminal")
    parser.add_argument("--scenario", default="alibaba2017_test")
    args = parser.parse_args()
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or not args.event_thresholds
        or len(set(args.event_thresholds)) != len(args.event_thresholds)
        or any(not 0 <= threshold <= 1 for threshold in args.event_thresholds)
    ):
        parser.error("Seeds and event thresholds must be unique and valid")
    files = [
        (
            mode,
            threshold,
            algorithm,
            seed,
            args.results_root
            / (
                f"fb-{mode}-threshold-{threshold:g}-{algorithm}-seed-{seed}-"
                f"{args.credit_mode}"
            )
            / "energy_success_bytes.csv",
        )
        for mode in args.feedback_modes
        for threshold in (
            args.event_thresholds if mode == "event" else [0.25]
        )
        for algorithm in args.algorithms
        for seed in args.seeds
    ]
    missing = [path for _, _, _, _, path in files if not path.is_file()]
    if missing:
        parser.error("Missing ablation results: " + ", ".join(map(str, missing)))
    try:
        verify_protocol([path for _, _, _, _, path in files])
        rows = summarize_files(files, args.scenario)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    if not rows:
        parser.error(f"No metrics found for scenario {args.scenario!r}")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
