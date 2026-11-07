"""Five-policy SimPy comparison plus causal-feedback ablations on held-out seeds.

Development experiments use synthetic tasks and one device/edge/cloud only.
Do NOT cite outputs as evidence of improvements on a public trace.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from mec_offloading.baselines import (
    DelayedGreedyComputeEdgePolicy,
    DelayedGreedyNearestDevicePolicy,
    EDFCloudPolicy,
    FIFOCloudPolicy,
    GreedyComputeEdgePolicy,
    GreedyNearestDevicePolicy,
    RandomCloudPolicy,
    RandomDevicePolicy,
    RandomEdgePolicy,
)
from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.actions import CloudSchedulingRule
from mec_offloading.evaluation.runner import _write_csv, aggregate_results, summarize_runs
from mec_offloading.experiment_metadata import configuration_hash
from mec_offloading.rl.credit import REWARD_VERSION, deadline_feasibility
from mec_offloading.rl.dqn import DQNAgent
from mec_offloading.rl.experiment_data import (
    BEST_TIER_DEADLINE_MULTIPLIER,
    BEST_TIER_DEADLINE_SLACK_S,
    DEFAULT_TRACE_ARRIVAL_COMPRESSION,
    compress_arrivals,
    load_trace_experiment,
    sample_tasks,
    topology_metadata,
)
from mec_offloading.rl.integrated import run_centralized, run_hierarchical
from mec_offloading.rl.observations import FEATURE_VERSION
from mec_offloading.rl.scenarios import SCENARIOS
from mec_offloading.rl.spaces import SPECS
from mec_offloading.rl.topology_variants import (
    HELD_OUT_VARIANTS,
    build_topology_variant,
)
from mec_offloading.simulation import SimPyBackend
from mec_offloading.trace_workloads import ALIBABA_2017_ADAPTER_VERSION


def baseline_bundles(seed: int):
    """Fresh random state per run, identical task/scenario for every policy."""
    return [
        ("random", RandomDevicePolicy(seed + 101), RandomEdgePolicy(seed + 202),
         RandomCloudPolicy(seed + 303)),
        ("greedy_fifo", GreedyNearestDevicePolicy(), GreedyComputeEdgePolicy(),
         FIFOCloudPolicy()),
        ("greedy_edf", GreedyNearestDevicePolicy(), GreedyComputeEdgePolicy(),
         EDFCloudPolicy()),
    ]


def delayed_greedy_bundle(
    feedback_mode: str, frequency: int, event_threshold: float
):
    coordinator = FeedbackCoordinator(
        feedback_mode, frequency, event_threshold=event_threshold
    )
    return (
        "greedy_delayed",
        DelayedGreedyNearestDevicePolicy(coordinator),
        DelayedGreedyComputeEdgePolicy(coordinator),
        EDFCloudPolicy(),
    )


def detailed_rows(label, results, seed, scenario, feedback_mode, event_threshold):
    for result in results:
        yield {
            "policy": label, "scenario": scenario, "feedback_mode": feedback_mode,
            "event_threshold": event_threshold,
            "seed": seed, "task_id": result.task_id,
            "route": "->".join(tier.value for tier in result.route),
            "arrival_time_s": result.arrival_time_s,
            "completion_time_s": result.completion_time_s,
            "execution_start_time_s": result.execution_start_time_s,
            "end_to_end_latency_s": result.end_to_end_latency_s,
            "deadline_met": result.deadline_met,
            "device_energy_j": result.device_energy_j,
            "queue_wait_s": result.queue_wait_s,
            "communication_delay_s": result.communication_delay_s,
            "communication_bytes": result.communication_bytes,
            "feedback_messages": result.feedback_messages,
            "feedback_bytes": result.feedback_bytes,
            "feedback_modeled_delay_s": result.feedback_modeled_delay_s,
            "feedback_age_s": result.feedback_age_s,
            "feedback_age_samples": result.feedback_age_samples,
            "failure_reason": result.failure_reason or "",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/models/integrated"))
    parser.add_argument(
        "--fixed-cloud-edf", action="store_true",
        help="Use EDF with the same device and edge checkpoints for a cloud ablation",
    )
    parser.add_argument("--cloud-rule-ablation", action="store_true",
                        help=("Compare learned cloud with EDF, SPT and FIFO at "
                              "fixed upstream weights"))
    parser.add_argument("--output", type=Path, default=Path("results/integrated-comparison"))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[10_001, 10_002, 10_003, 10_004, 10_005])
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--arrival-compression", type=float)
    parser.add_argument("--test-deadline-floor", type=float,
                        help="Override the best-tier deadline floor at test time only")
    parser.add_argument("--source", choices=["synthetic", "alibaba2017"],
                        default="synthetic")
    parser.add_argument(
        "--deadline-policy",
        choices=["fixed-5s", "local-relative-2x", "best-tier-calibrated"],
        default="fixed-5s",
        help="Alibaba-only deadline assumption used for training and held-out evaluation",
    )
    parser.add_argument("--trace-path", type=Path,
                        default=Path("data/raw/batch_task.csv"))
    parser.add_argument("--trace-instances", type=Path,
                        default=Path("data/raw/batch_instance.csv"))
    parser.add_argument("--topology-config", type=Path,
                        default=Path("configs/alibaba-multi-topology.yaml"))
    parser.add_argument(
        "--max-trace-tasks",
        type=int,
        default=5_000,
        help="maximum accepted trace tasks retained (source CSVs are fully scanned)",
    )
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=["nominal", "high_load", "congested_network"])
    parser.add_argument("--feedback-modes", nargs="+",
                        choices=["none", "periodic", "frequent", "event"],
                        default=None)
    parser.add_argument("--feedback", choices=["none", "periodic", "frequent", "event"],
                        help="Legacy one-mode option; overrides --feedback-modes")
    parser.add_argument("--feedback-frequency", type=int, default=5)
    parser.add_argument("--event-thresholds", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5])
    args = parser.parse_args()
    if args.tasks is None:
        args.tasks = 5_000 if args.source == "alibaba2017" else 100
    if (
        args.tasks <= 0
        or args.feedback_frequency <= 0
        or not args.event_thresholds
        or len(set(args.event_thresholds)) != len(args.event_thresholds)
        or any(not 0 <= threshold <= 1 for threshold in args.event_thresholds)
        or args.max_trace_tasks < 3
        or (
            args.arrival_compression is not None
            and (
                not math.isfinite(args.arrival_compression)
                or args.arrival_compression <= 0
            )
        )
    ):
        parser.error("tasks and feedback frequency must be > 0; trace limit must be >= 3")
    if args.source != "alibaba2017" and args.deadline_policy != "fixed-5s":
        parser.error("--deadline-policy is only applicable with --source alibaba2017")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Evaluation seeds must be unique")
    modes = [args.feedback] if args.feedback is not None else []

    metadata_file = args.models / "metadata.json"
    if not metadata_file.is_file():
        parser.error("No trained model metadata. Run train_integrated_rl.py first")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if args.arrival_compression is None:
        args.arrival_compression = metadata.get(
            "arrival_compression",
            DEFAULT_TRACE_ARRIVAL_COMPRESSION if args.source == "alibaba2017" else 1.0,
        )
    if args.arrival_compression <= 0:
        parser.error("arrival compression must be > 0")
    if metadata.get("feature_version") != FEATURE_VERSION:
        parser.error("Incompatible observation version; retrain integrated agents")
    if metadata.get("reward_version") != REWARD_VERSION:
        parser.error("Incompatible reward version; train matching integrated agents")
    if metadata.get("arrival_compression", 1.0) != args.arrival_compression:
        parser.error("Evaluation arrival compression differs from training")
    if args.feedback is not None and metadata.get("feedback") != args.feedback:
        parser.error("Evaluate the checkpoint with the same feedback mode it was trained on")
    if not modes:
        modes = list(dict.fromkeys(args.feedback_modes or (
            [metadata.get("feedback", "periodic")]
            if args.source == "alibaba2017"
            else ["none", "periodic", "frequent"]
        )))
    trained_source = metadata.get("workload_source", "synthetic")
    if trained_source != args.source:
        parser.error(
            f"Checkpoint source is {trained_source!r}, but evaluation source is "
            f"{args.source!r}; train matching checkpoints first"
        )
    reserved = set(metadata.get("training_seeds", [])) | set(metadata.get("validation_seeds", []))
    overlap = reserved.intersection(args.seeds)
    if overlap:
        parser.error(f"Data leakage: evaluation seeds overlap train/validation: {sorted(overlap)}")
    freeze_cloud = (
        args.fixed_cloud_edf
        or metadata.get("cloud_scheduler", "learned-DQN") != "learned-DQN"
    )
    agent_names = ["device", "edge", "centralized"]
    if not freeze_cloud:
        agent_names.append("cloud")
    agents = {name: DQNAgent.load(args.models / f"{name}.pt") for name in agent_names}
    if args.cloud_rule_ablation and freeze_cloud:
        parser.error("--cloud-rule-ablation requires a learned cloud checkpoint")
    for name, agent in agents.items():
        spec = SPECS[name]
        if (agent.obs_dim, agent.n_actions) != (spec.observation_dim, len(spec.actions)):
            parser.error(f"Checkpoint schema/action mismatch for {name}")

    if args.test_deadline_floor is not None and (
        args.source != "alibaba2017"
        or args.deadline_policy != "best-tier-calibrated"
        or not math.isfinite(args.test_deadline_floor)
        or args.test_deadline_floor <= 0
    ):
        parser.error(
            "--test-deadline-floor requires Alibaba best-tier-calibrated and a positive floor"
        )
    trace_data = None
    if args.source == "alibaba2017":
        try:
            trace_data = load_trace_experiment(
                args.trace_path,
                args.trace_instances,
                args.topology_config,
                max_tasks=args.max_trace_tasks,
                deadline_policy=args.deadline_policy,
            )
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"Could not prepare Alibaba 2017 RL workload: {error}")
        if metadata.get("topology") != topology_metadata(trace_data.simulation_config):
            parser.error("Checkpoint topology differs from the selected trace topology")
        if (
            metadata.get("workload", {}).get("trace_adapter_version")
            != ALIBABA_2017_ADAPTER_VERSION
        ):
            parser.error(
                "Checkpoint uses an incompatible Alibaba task adapter; retrain "
                "with representative-instance task work"
            )
        if metadata.get("deadline_policy", "fixed-5s") != args.deadline_policy:
            parser.error(
                f"Checkpoint deadline policy is "
                f"{metadata.get('deadline_policy', 'fixed-5s')!r}, but evaluation "
                f"requested {args.deadline_policy!r}; train matching checkpoints first"
            )
        trained_workload = metadata.get("workload", {})
        for field in (
            "topology_sha256",
            "trace_sha256",
            "instances_sha256",
            "selected_tasks",
        ):
            if trained_workload.get(field) != trace_data.metadata.get(field):
                parser.error(
                    f"Evaluation {field} differs from training; use the same trace "
                    "files and selected subset"
                )
    else:
        trace_data = None

    aggregates, details = [], []
    cloud_decisions = []
    scenario_names = (
        [
            "alibaba2017_test",
            *[
                f"alibaba2017_test:{variant.name}"
                for variant in HELD_OUT_VARIANTS
            ],
        ]
        if trace_data is not None else list(args.scenarios)
    )
    for scenario_name in scenario_names:
        scenario = SCENARIOS.get(scenario_name)
        topology_name = scenario_name.partition(":")[2]
        topology_variant = next(
            (variant for variant in HELD_OUT_VARIANTS
             if variant.name == topology_name),
            None,
        )
        for seed in args.seeds:
            if trace_data is None:
                if scenario is None:
                    raise RuntimeError(f"Unknown synthetic scenario {scenario_name}")
                tasks = compress_arrivals(
                    scenario.tasks(args.tasks, seed), args.arrival_compression
                )
                make_backend = scenario.backend
            else:
                tasks = compress_arrivals(
                    sample_tasks(trace_data.test, args.tasks, seed),
                    args.arrival_compression,
                )
                variant_config = trace_data.simulation_config
                if args.test_deadline_floor is not None:
                    topology = variant_config.topology
                    reference_capacity = max(
                        variant_config.cloud_cpu_capacity_mips,
                        *(node.cpu_capacity_mips for node in (*topology.devices, *topology.edges)),
                    )
                    tasks = [
                        replace(task, deadline_s=max(
                            args.test_deadline_floor,
                            BEST_TIER_DEADLINE_MULTIPLIER * task.required_mi / reference_capacity
                            + BEST_TIER_DEADLINE_SLACK_S,
                        ))
                        for task in tasks
                    ]
                if topology_variant is not None:
                    variant_config = build_topology_variant(
                        variant_config, topology_variant
                    )

                def make_backend(config=variant_config):
                    return SimPyBackend(config=config)

            feasibility = (
                deadline_feasibility(tasks, make_backend())
                if trace_data is not None
                else {task.task_id: True for task in tasks}
            )
            experiments = []
            for label, device, edge, cloud in baseline_bundles(seed):
                results = make_backend().run_episode(
                    tasks, device, edge, cloud, seed=seed,
                )
                experiments.append((label, results, "n/a", ""))
            cent_results, _ = run_centralized(
                tasks, agents["centralized"], seed=seed, backend=make_backend(),
            )
            experiments.append(("centralized_rl", cent_results, "n/a", ""))
            for mode in modes:
                thresholds = args.event_thresholds if mode == "event" else [0.25]
                for event_threshold in thresholds:
                    condition_experiments = []
                    if mode == modes[0] and event_threshold == thresholds[0]:
                        condition_experiments.extend(experiments)
                    label, device_policy, edge_policy, cloud_policy = delayed_greedy_bundle(
                        mode, args.feedback_frequency, event_threshold
                    )
                    delayed_results = make_backend().run_episode(
                        tasks, device_policy, edge_policy, cloud_policy, seed=seed,
                    )
                    condition_experiments.append(
                        (label, delayed_results, mode, event_threshold)
                    )
                    hier_results, logs = run_hierarchical(
                        tasks, agents, seed=seed, backend=make_backend(),
                        feedback_mode=mode, feedback_frequency=args.feedback_frequency,
                        event_threshold=event_threshold, freeze_cloud=freeze_cloud,
                    )
                    condition_experiments.append(
                        (f"hierarchical_{mode}", hier_results, mode, event_threshold)
                    )
                    if args.cloud_rule_ablation:
                        for rule in CloudSchedulingRule:
                            fixed_results, _ = run_hierarchical(
                                tasks, agents, seed=seed, backend=make_backend(),
                                feedback_mode=mode,
                                feedback_frequency=args.feedback_frequency,
                                event_threshold=event_threshold,
                                fixed_cloud_rule=rule,
                            )
                            condition_experiments.append((
                                f"hierarchical_{mode}_fixed_{rule.name.lower()}",
                                fixed_results, mode, event_threshold,
                            ))
                            cloud_decisions.append({
                                "scenario": scenario_name, "seed": seed,
                                "feedback_mode": mode,
                                "event_threshold": event_threshold,
                                "rule": rule.name,
                                "contested_dispatches": (
                                    logs["cloud"].contested_actions.get(rule, 0)
                                ),
                            })

                    for label, results, feedback, threshold in condition_experiments:
                        row = {
                            "scenario": scenario_name,
                            "feedback_mode": feedback,
                            "event_threshold": threshold,
                        }
                        row.update(aggregate_results(label, results, seed=seed))
                        if trace_data is not None:
                            feasible_results = [
                                result for result in results if feasibility[result.task_id]
                            ]
                            row.update({
                                "feasible_task_count": len(feasible_results),
                                "infeasible_task_count": len(results) - len(feasible_results),
                                "feasible_deadline_success_rate": (
                                    sum(result.deadline_met for result in feasible_results)
                                    / len(feasible_results)
                                    if feasible_results else 0.0
                                ),
                            })
                        aggregates.append(row)
                        details.extend(detailed_rows(
                            label, results, seed, scenario_name, feedback, threshold
                        ))
            print(f"Scenario={scenario_name}, seed={seed}: comparison complete")

    args.output.mkdir(parents=True, exist_ok=True)
    comparison_config = {
        "source": args.source,
        "deadline_policy": args.deadline_policy,
        "arrival_compression": args.arrival_compression,
        "test_seeds": args.seeds,
        "tasks": args.tasks,
        "scenarios": scenario_names,
        "workload_fingerprints": (
            {
                key: trace_data.metadata.get(key)
                for key in ("trace_sha256", "instances_sha256", "topology_sha256")
            }
            if trace_data is not None else None
        ),
        "fixed_cloud_edf": args.fixed_cloud_edf,
        "cloud_rule_ablation": args.cloud_rule_ablation,
        "feedback_modes": modes,
        "feedback_frequency": args.feedback_frequency,
        "event_thresholds": args.event_thresholds,
        "training_configuration_hash": metadata.get("configuration_hash"),
        "training_feedback": metadata.get("feedback"),
        "training_event_threshold": metadata.get("event_threshold", 0.25),
        "training_algorithm": metadata.get("dqn", {}).get("algorithm", "double-dqn"),
        "test_deadline_floor": args.test_deadline_floor,
    }
    config_hash = configuration_hash(comparison_config)
    for row in aggregates:
        row["configuration_hash"] = config_hash
    for row in details:
        row["configuration_hash"] = config_hash
    for row in cloud_decisions:
        row["configuration_hash"] = config_hash
    statistics = []
    for scenario_name in scenario_names:
        scenario_rows = [row for row in aggregates if row["scenario"] == scenario_name]
        extra_metric_keys = (
            ("feasible_task_count", "infeasible_task_count",
             "feasible_deadline_success_rate")
            if trace_data is not None else ()
        )
        conditions = sorted({
            (str(row["feedback_mode"]), str(row["event_threshold"]))
            for row in scenario_rows
        })
        for feedback_mode, event_threshold in conditions:
            condition_rows = [
                row for row in scenario_rows
                if str(row["feedback_mode"]) == feedback_mode
                and str(row["event_threshold"]) == event_threshold
            ]
            for row in summarize_runs(
                condition_rows, extra_metric_keys=extra_metric_keys
            ):
                statistics.append({
                    "scenario": scenario_name,
                    "feedback_mode": feedback_mode,
                    "event_threshold": event_threshold,
                    **row,
                    "configuration_hash": config_hash,
                })
    _write_csv(args.output / "task_results.csv", details)
    _write_csv(args.output / "aggregate_metrics.csv", aggregates)
    if cloud_decisions:
        _write_csv(args.output / "cloud_contested_actions.csv", cloud_decisions)
    _write_csv(args.output / "aggregate_statistics.csv", statistics)
    _write_csv(
        args.output / "energy_success_bytes.csv",
        [
            {
                "scenario": row["scenario"],
                "policy": row["policy"],
                "seed": row["seed"],
                "deadline_success_rate": row["deadline_success_rate"],
                "p95_latency_s": row["p95_latency_s"],
                "mean_device_energy_j": row["mean_device_energy_j"],
                "communication_bytes": row["communication_bytes"],
                "feedback_bytes": row["feedback_bytes"],
                "mean_feedback_age_s": row["mean_feedback_age_s"],
                "total_bytes": row["communication_bytes"] + row["feedback_bytes"],
                "configuration_hash": config_hash,
            }
            for row in aggregates
        ],
    )
    (args.output / "metadata.json").write_text(json.dumps({
        "source": (
            "Synthetic development workload"
            if trace_data is None else "Alibaba Cluster Trace 2017"
        ),
        "workload_source": args.source,
        "deadline_policy": (
            trace_data.metadata["task_assumptions"]["deadline_policy"]
            if trace_data is not None else "synthetic-scenario"
        ),
        "workload": trace_data.metadata if trace_data is not None else {
            "synthetic_scenarios": args.scenarios,
            "topology": topology_metadata(SCENARIOS[args.scenarios[0]].backend().config),
        },
        "test_seeds": args.seeds,
        "training_seed": metadata.get("training_seed"),
        "checkpoint_cloud_scheduler": metadata.get("cloud_scheduler"),
        "cloud_rule_ablation": args.cloud_rule_ablation,
        "train_seeds": metadata.get("training_seeds", []),
        "validation_seeds": metadata.get("validation_seeds", []),
        "task_count_per_scenario_seed": (
            min(args.tasks, len(trace_data.test)) if trace_data is not None else args.tasks
        ),
        "arrival_compression": args.arrival_compression,
        "test_deadline_floor": args.test_deadline_floor,
        "scenarios": scenario_names,
        "feedback_modes": modes,
        "feedback": metadata.get("feedback"),
        "feedback_frequency": args.feedback_frequency,
        "event_thresholds": args.event_thresholds,
        "training_event_threshold": metadata.get("event_threshold", 0.25),
        "training_algorithm": metadata.get("dqn", {}).get("algorithm", "double-dqn"),
        "configuration": comparison_config,
        "configuration_hash": config_hash,
        "cloud_scheduler": (
            "EDF-fixed" if freeze_cloud else "learned-DQN"
        ),
        "feasibility_definition": (
            "Any no-queue local, edge, or cloud route can meet deadline"
            if trace_data is not None else "not applicable"
        ),
        "training_feature_version": FEATURE_VERSION,
        "simulator": (
            "SimPyBackend; one execution slot per device/edge and one cloud worker; "
            "queue-wait estimates use active remaining and queued MI"
        ),
        "comparison_policies": sorted({row["policy"] for row in aggregates}),
        "feedback_note": "Causal decision-time arrival, no control-plane network contention",
        "note": "No claim of real-trace improvement; additional real topology needed",
    }, indent=2), encoding="utf-8")
    print("Comparison and ablations written to", args.output)


if __name__ == "__main__":
    main()
