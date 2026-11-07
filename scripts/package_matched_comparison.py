"""Package the final six-policy comparison from existing Alibaba replay outputs."""

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
from statistics import mean, stdev

from mec_offloading.experiment_metadata import configuration_hash
from mec_offloading.rl.credit import REWARD_VERSION
from mec_offloading.rl.observations import FEATURE_VERSION
from mec_offloading.trace_workloads import ALIBABA_2017_ADAPTER_VERSION

SEEDS = (42, 43, 44, 45, 46)
NOISE = (10001, 10002, 10003, 10004, 10005)
SCENARIO = "alibaba2017_test"
BASELINES = ("random", "greedy_fifo", "greedy_edf")
METRICS = (
    "deadline_success_rate", "feasible_deadline_success_rate", "mean_latency_s",
    "median_latency_s", "p95_latency_s", "mean_device_energy_j",
    "communication_bytes", "feedback_bytes", "mean_feedback_age_s",
)
POLICIES = (*BASELINES, "centralized_dqn", "hierarchical_dqn", "hierarchical_double_dqn")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(stream, rows):
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/feedback-ablation"))
    parser.add_argument("--models-root", type=Path, default=Path("artifacts/models/feedback-ablation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aggregate, detailed, sources = [], [], []
    baseline_reference = {}
    task_reference = None
    shared_train = shared_eval = shared_topology = None
    for algorithm in ("dqn", "double-dqn"):
        for training_seed in SEEDS:
            run = f"fb-frequent-threshold-0.25-{algorithm}-seed-{training_seed}-next-decision"
            source = args.results_root / run
            model = args.models_root / run
            training = read_json(model / "metadata.json")
            evaluation = read_json(source / "metadata.json")
            train_config = training["configuration"]
            eval_config = evaluation["configuration"]
            require(
                configuration_hash(train_config) == training["configuration_hash"]
                == eval_config["training_configuration_hash"]
                and configuration_hash(eval_config) == evaluation["configuration_hash"],
                f"Inconsistent configuration hashes for {run}",
            )
            require(
                training["training_seed"] == evaluation["training_seed"] == training_seed
                and training["dqn"]["algorithm"] == evaluation["training_algorithm"] == algorithm
                and training["feature_version"] == evaluation["training_feature_version"]
                == FEATURE_VERSION
                and training["reward_version"] == REWARD_VERSION
                and training["workload"]["trace_adapter_version"]
                == evaluation["workload"]["trace_adapter_version"] == ALIBABA_2017_ADAPTER_VERSION
                and training["credit_mode"] == "next-decision"
                and training["feedback"] == evaluation["feedback"] == "frequent"
                and training["cloud_scheduler"] == evaluation["cloud_scheduler"] == "learned-DQN"
                and eval_config["source"] == "alibaba2017"
                and eval_config["test_deadline_floor"] is None
                and evaluation["task_count_per_scenario_seed"] == 579
                and tuple(evaluation["test_seeds"]) == NOISE
                and not set(NOISE).intersection(training["training_seeds"] + training["validation_seeds"]),
                f"Unexpected checkpoint or evaluation protocol for {run}",
            )
            topology = training["topology"]
            require(
                len(topology["devices"]) == 24 and len(topology["edges"]) == 3,
                f"Unexpected topology for {run}",
            )
            checkpoints = [f"{tier}.pt" for tier in ("device", "edge", "cloud", "centralized")]
            require(all((model / name).is_file() for name in checkpoints),
                    f"Missing checkpoint for {run}")
            comparable_train = {k: v for k, v in train_config.items() if k != "algorithm"}
            comparable_eval = {
                k: v for k, v in eval_config.items()
                if k not in ("training_algorithm", "training_configuration_hash")
            }
            if shared_train is None:
                shared_train, shared_eval, shared_topology = comparable_train, comparable_eval, topology
            require(
                (comparable_train, comparable_eval, topology)
                == (shared_train, shared_eval, shared_topology),
                f"Mixed protocols in {run}",
            )
            sources.append({
                "run": run, "training_seed": training_seed, "algorithm": algorithm,
                "observation_version": training["feature_version"],
                "reward_version": training["reward_version"],
                "training_configuration_hash": training["configuration_hash"],
                "evaluation_configuration_hash": evaluation["configuration_hash"],
                "episodes_completed": training["episodes_completed"],
                "selected_episodes": training["selected_episodes"],
                "checkpoints": checkpoints,
            })
            mapping = {"hierarchical_frequent": "hierarchical_" + algorithm.replace("-", "_")}
            if algorithm == "dqn":
                mapping["centralized_rl"] = "centralized_dqn"
            for file, output in (("task_results.csv", detailed), ("aggregate_metrics.csv", aggregate)):
                with (source / file).open(newline="", encoding="utf-8") as stream:
                    rows = [row for row in csv.DictReader(stream) if row["scenario"] == SCENARIO]
                for original in (*BASELINES, *mapping):
                    for noise_seed in NOISE:
                        group = [
                            row for row in rows
                            if row["policy"] == original and int(row["seed"]) == noise_seed
                        ]
                        per_task = file == "task_results.csv"
                        require(len(group) == (579 if per_task else 1),
                                f"Unexpected row count in {run}/{file}/{original}/{noise_seed}")
                        if per_task:
                            tasks = sorted((row["task_id"], row["arrival_time_s"]) for row in group)
                            require(len({task for task, _ in tasks}) == 579, "Duplicate task IDs")
                            if task_reference is None:
                                task_reference = tasks
                            require(tasks == task_reference, f"Unmatched task set in {run}")
                        else:
                            require(int(group[0]["task_count"]) == 579, "Unmatched task count")
                        if original in BASELINES:
                            key = file, original, noise_seed
                            comparable = [
                                {k: v for k, v in row.items() if k != "configuration_hash"}
                                for row in group
                            ]
                            if key in baseline_reference:
                                require(comparable == baseline_reference[key],
                                        f"Baseline result differs in {run}/{original}/{noise_seed}")
                                continue
                            baseline_reference[key] = comparable
                        output.extend({
                            **row,
                            "policy": mapping.get(original, original),
                            "training_seed": "" if original in BASELINES else training_seed,
                            "source_run": run,
                        } for row in group)

    summary = []
    for policy in POLICIES:
        rows = [row for row in aggregate if row["policy"] == policy]
        trained = policy not in BASELINES
        require(len(rows) == (25 if trained else 5), f"Incomplete results for {policy}")
        for metric in METRICS:
            values = [
                mean(float(row[metric]) for row in rows if row["training_seed"] == seed)
                for seed in SEEDS
            ] if trained else [float(row[metric]) for row in rows]
            summary.append({
                "policy": policy, "metric": metric, "mean": mean(values),
                "training_seed_sd": stdev(values) if trained else "",
                "baseline_noise_seed_sd": "" if trained else stdev(values),
                "training_seeds": len(SEEDS) if trained else 0,
                "evaluation_runs": len(rows),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("aggregate_metrics.csv", aggregate), ("summary.csv", summary)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            write_csv(stream, rows)
    with (args.output / "task_results.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                write_csv(stream, detailed)
    (args.output / "metadata.json").write_text(json.dumps({
        "scenario": SCENARIO, "tasks_per_run": 579,
        "training_seeds": SEEDS, "evaluation_noise_seeds": NOISE,
        "policies": POLICIES, "task_rows": len(detailed), "aggregate_rows": len(aggregate),
        "centralized_algorithm": "dqn",
        "baseline_handling": "Identical outputs checked across all source bundles and kept once per noise seed.",
        "training_configuration_except_algorithm": shared_train,
        "evaluation_configuration_except_algorithm_and_training_hash": shared_eval,
        "topology": shared_topology, "source_runs": sources,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(detailed)} task rows and {len(aggregate)} evaluation rows to {args.output}")


if __name__ == "__main__":
    main()
