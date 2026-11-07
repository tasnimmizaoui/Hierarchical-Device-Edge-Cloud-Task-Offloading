"""Paired bootstrap for deadline-success differences in task-level result CSVs."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def as_bool(value: str) -> float:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return 1.0
    if normalized in {"false", "0", "no"}:
        return 0.0
    raise ValueError(f"Invalid deadline_met value: {value!r}")


def paired_differences(
    input_path: Path, left: str, right: str, scenario: str | None = None
) -> list[float]:
    results: dict[tuple[str, str, str], dict[str, float]] = {}
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"task_id", "policy", "deadline_met"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError(
                f"{input_path} must contain task_id, policy, and deadline_met"
            )
        for row in reader:
            if scenario is not None and row.get("scenario") != scenario:
                continue
            key = (
                row.get("scenario", ""),
                row.get("seed", ""),
                row["task_id"],
            )
            results.setdefault(key, {})[row["policy"]] = as_bool(
                row["deadline_met"]
            )
    return [
        policies[left] - policies[right]
        for policies in results.values()
        if left in policies and right in policies
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", type=Path, nargs="+",
        help="task_results.csv containing task_id, policy, and deadline_met columns",
    )
    parser.add_argument("--left", default="hierarchical_periodic")
    parser.add_argument("--right", default="greedy_edf")
    parser.add_argument("--scenario")
    parser.add_argument("--replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be > 0")

    differences = []
    for input_path in args.inputs:
        try:
            differences.extend(
                paired_differences(
                    input_path, args.left, args.right, args.scenario
                )
            )
        except ValueError as error:
            parser.error(str(error))
    if not differences:
        parser.error(f"No paired tasks found for {args.left!r} and {args.right!r}")
    generator = random.Random(args.seed)
    bootstrap_means = [
        sum(generator.choices(differences, k=len(differences))) / len(differences)
        for _ in range(args.replicates)
    ]
    bootstrap_means.sort()
    lower = bootstrap_means[int(.025 * (args.replicates - 1))]
    upper = bootstrap_means[int(.975 * (args.replicates - 1))]
    print(f"Matched task-seed pairs: {len(differences)}")
    print(f"Discordant outcomes: {sum(value != 0 for value in differences)}")
    print(f"Mean success-rate difference ({args.left} - {args.right}): "
          f"{sum(differences) / len(differences):.6f}")
    print(f"95% paired bootstrap CI: [{lower:.6f}, {upper:.6f}]")


if __name__ == "__main__":
    main()
