"""Train DQN agents on synthetic scenarios or held-out trace/topology splits."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from mec_offloading.experiment_metadata import configuration_hash
from mec_offloading.rl.credit import (
    REWARD_VERSION,
    RewardWeights,
    deadline_feasibility,
)
from mec_offloading.rl.experiment_data import (
    DEFAULT_TRACE_ARRIVAL_COMPRESSION,
    compress_arrivals,
    load_trace_experiment,
    offered_load,
    sample_tasks,
    topology_metadata,
)
from mec_offloading.rl.integrated import (
    make_agents,
    run_centralized,
    run_hierarchical,
    team_reward,
    update_dqn_from_episode,
)
from mec_offloading.rl.observations import FEATURE_VERSION
from mec_offloading.rl.scenarios import SCENARIOS
from mec_offloading.rl.topology_variants import (
    HELD_OUT_VARIANTS,
    TRAINING_VARIANTS,
    build_topology_variant,
    shuffled_variant_schedule,
)
from mec_offloading.simulation import SimPyBackend


def mean_task_reward(tasks, results, feasibility=None,
                     weights: RewardWeights | None = None) -> float:
    outcomes = {r.task_id: r for r in results}
    if len(outcomes) != len(tasks):
        raise RuntimeError("Missing completed tasks in SimPy run")
    return float(np.mean([
        team_reward(
            task,
            outcomes[task.task_id],
            weights,
            deadline_feasible=(
                feasibility.get(task.task_id, True) if feasibility is not None else True
            ),
        )
        for task in tasks
    ]))


def score_validation(agents, cases, feedback: str,
                     frequency: int, *, freeze_cloud: bool,
                     weights: RewardWeights,
                     event_threshold: float) -> tuple[float, float, dict]:
    """Evaluate greedily, without replay updates or exploration."""
    hierarchical, centralized = [], []
    metrics = {"hierarchical_deadline": [], "centralized_deadline": [],
               "hierarchical_feasible_deadline": [], "centralized_feasible_deadline": [],
               "hierarchical_latency": [], "centralized_latency": [],
               "hierarchical_median_latency": [], "centralized_median_latency": [],
               "hierarchical_p95_latency": [], "centralized_p95_latency": []}
    for seed, tasks, backend in cases:
        feasibility = deadline_feasibility(tasks, backend)
        hier, _ = run_hierarchical(
            tasks, agents, seed=seed, backend=backend,
            feedback_mode=feedback, feedback_frequency=frequency,
            event_threshold=event_threshold,
            training=False, freeze_cloud=freeze_cloud,
        )
        cent, _ = run_centralized(
            tasks, agents["centralized"], seed=seed,
            backend=backend, training=False,
        )
        hierarchical.append(mean_task_reward(tasks, hier, feasibility, weights))
        centralized.append(mean_task_reward(tasks, cent, feasibility, weights))
        for label, results in (("hierarchical", hier), ("centralized", cent)):
            metrics[f"{label}_deadline"].append(np.mean([r.deadline_met for r in results]))
            feasible_results = [
                result for result in results if feasibility[result.task_id]
            ]
            metrics[f"{label}_feasible_deadline"].append(
                np.mean([r.deadline_met for r in feasible_results])
                if feasible_results else 0.0
            )
            metrics[f"{label}_latency"].append(np.mean([
                r.end_to_end_latency_s for r in results]))
            latencies = [r.end_to_end_latency_s for r in results]
            metrics[f"{label}_median_latency"].append(float(np.median(latencies)))
            metrics[f"{label}_p95_latency"].append(
                float(np.percentile(latencies, 95))
            )
    return float(np.mean(hierarchical)), float(np.mean(centralized)), {
        key: float(np.mean(values)) for key, values in metrics.items()
    }


def save_team(agents, output: Path, members: tuple[str, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in members:
        agents[name].save(output / f"{name}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", choices=["synthetic", "alibaba2017"],
                        default="synthetic")
    parser.add_argument(
        "--topology-augmentation",
        choices=["none", "sweep"],
        default="none",
        help="For Alibaba training, sample episodes across the nine training topologies",
    )
    parser.add_argument("--arrival-compression", type=float)
    parser.add_argument(
        "--credit-mode",
        choices=["terminal", "next-decision"],
        default="terminal",
        help="Link each transition to terminal credit or the same agent's next decision",
    )
    parser.add_argument(
        "--deadline-policy",
        choices=["fixed-5s", "local-relative-2x", "best-tier-calibrated"],
        default="fixed-5s",
        help="Alibaba-only deadline assumption; non-default policies are synthetic",
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learn-cloud",
        action="store_true",
        help="train the cloud DQN; default uses fixed EDF while device/edge learn",
    )
    parser.add_argument("--feedback", choices=["none", "periodic", "frequent", "event"],
                        default="periodic")
    parser.add_argument("--feedback-frequency", type=int, default=5)
    parser.add_argument("--event-threshold", type=float, default=0.25)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=None)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--validation-tasks", type=int, default=578)
    parser.add_argument("--patience", type=int, default=8,
                        help="Stop after this many non-improving validation checks")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--exploration-schedule",
                        choices=["linear-episode-75pct", "exponential-decisions"],
                        default="linear-episode-75pct")
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibrate-on-validation",
                             dest="calibrate_on_validation", action="store_true")
    calibration.add_argument("--configured-compression",
                             dest="calibrate_on_validation", action="store_false")
    parser.set_defaults(calibrate_on_validation=None)
    parser.add_argument("--validation-selection", choices=["reward", "deadline-p95"],
                        default="deadline-p95")
    parser.add_argument("--contiguous-trace-windows", action="store_true",
                        help="Sample contiguous training tasks at a random chronological offset")
    parser.add_argument("--algorithm", choices=["dqn", "double-dqn"],
                        default="double-dqn")
    parser.add_argument("--reward-weights", type=float, nargs=4,
                        metavar=("LATENCY", "ENERGY", "DEADLINE", "FEEDBACK"),
                        default=(0.35, 0.25, 0.35, 0.05))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/integrated"))
    args = parser.parse_args()
    if args.calibrate_on_validation is None:
        args.calibrate_on_validation = (
            args.source == "alibaba2017" and args.arrival_compression is None
        )
    if args.calibrate_on_validation and (
        args.source != "alibaba2017" or args.arrival_compression is not None
    ):
        parser.error("Validation calibration requires Alibaba and no explicit compression")
    if args.arrival_compression is None:
        args.arrival_compression = (
            DEFAULT_TRACE_ARRIVAL_COMPRESSION
            if args.source == "alibaba2017"
            else 1.0
        )
    if min(args.episodes, args.tasks, args.feedback_frequency, args.batch_size,
           args.validation_every, args.validation_tasks, args.patience) <= 0:
        parser.error("All episode, task, batch, frequency and validation counts must be > 0")
    if not 0 <= args.event_threshold <= 1:
        parser.error("--event-threshold must be in [0, 1]")
    if any(not np.isfinite(weight) or weight < 0 for weight in args.reward_weights):
        parser.error("--reward-weights must be finite and nonnegative")
    if not np.isclose(sum(args.reward_weights), 1.0):
        parser.error("--reward-weights must sum to 1")
    reward_weights = RewardWeights(*args.reward_weights)
    if not np.isfinite(args.arrival_compression) or args.arrival_compression <= 0:
        parser.error("--arrival-compression must be finite and > 0")
    if args.source != "alibaba2017" and args.deadline_policy != "fixed-5s":
        parser.error("--deadline-policy is only applicable with --source alibaba2017")
    if args.topology_augmentation != "none" and args.source != "alibaba2017":
        parser.error("--topology-augmentation requires --source alibaba2017")
    if args.max_trace_tasks < 3:
        parser.error("--max-trace-tasks must be >= 3 to populate all trace splits")
    scenarios = [SCENARIOS[name] for name in (
        args.scenarios or ["nominal", "high_load", "congested_network"]
    )]
    validation_seeds = (
        args.validation_seeds
        if args.validation_seeds is not None
        else [100_001 + i for i in range(5)]
    )
    training_seeds = {args.seed + i for i in range(args.episodes)}
    if training_seeds.intersection(validation_seeds):
        parser.error("Validation seeds must be disjoint from all training seeds")
    if len(set(validation_seeds)) != len(validation_seeds):
        parser.error("Validation seeds must be unique")

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
        if args.calibrate_on_validation:
            args.arrival_compression = offered_load(trace_data.validation, 12_500.0) / 0.75
        validation_tasks = sample_tasks(
            trace_data.validation, args.validation_tasks, validation_seeds[0]
        )
        validation_tasks = compress_arrivals(
            validation_tasks, args.arrival_compression
        )
        validation_cases = [
            (seed, validation_tasks, SimPyBackend(config=trace_data.simulation_config))
            for seed in validation_seeds
        ]
    else:
        trace_data = None
        validation_cases = [
            (
                seed,
                compress_arrivals(
                    scenario.tasks(args.validation_tasks, seed),
                    args.arrival_compression,
                ),
                scenario.backend(),
            )
            for scenario in scenarios
            for seed in validation_seeds
        ]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    print("Reward is negative normalized cost; closer to zero is better.")
    gamma = 0.9 if args.credit_mode == "next-decision" else 0.94
    agents = make_agents(
        args.seed, batch_size=args.batch_size, gamma=gamma, algorithm=args.algorithm
    )
    freeze_cloud = not args.learn_cloud
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_hier = float("-inf")
    best_central = float("-inf")
    best_keys = {}
    best_episodes = {}
    stale_validation_checks = {"hierarchical": 0, "centralized": 0}
    decision_history: list[dict[str, object]] = []
    experiment_config = {
        "episodes": args.episodes,
        "tasks_per_episode": args.tasks,
        "source": args.source,
        "deadline_policy": args.deadline_policy,
        "topology_augmentation": args.topology_augmentation,
        "training_window": (
            "contiguous" if args.contiguous_trace_windows else "sampled"
        ),
        "max_trace_tasks": args.max_trace_tasks,
        "scenarios": args.scenarios or [
            "nominal", "high_load", "congested_network"
        ],
        "learn_cloud": args.learn_cloud,
        "gamma": gamma,
        "arrival_compression": args.arrival_compression,
        "calibrate_on_validation": args.calibrate_on_validation,
        "algorithm": args.algorithm,
        "credit_mode": args.credit_mode,
        "exploration_schedule": args.exploration_schedule,
        "epsilon_start": 0.90,
        "epsilon_end": 0.05,
        "epsilon_decay_fraction": 0.75,
        "batch_size": args.batch_size,
        "replay_capacity": 8_000,
        "validation_tasks": args.validation_tasks,
        "validation_seeds": validation_seeds,
        "validation_selection": args.validation_selection,
        "validation_every": args.validation_every,
        "patience": args.patience,
        "reward_weights": args.reward_weights,
        "feedback": args.feedback,
        "feedback_frequency": args.feedback_frequency,
        "event_threshold": args.event_threshold,
        "workload_fingerprints": (
            {
                key: trace_data.metadata.get(key)
                for key in ("trace_sha256", "instances_sha256", "topology_sha256")
            }
            if trace_data is not None else None
        ),
    }
    config_hash = configuration_hash(experiment_config)
    topology_schedule = (
        shuffled_variant_schedule(TRAINING_VARIANTS, args.episodes, args.seed)
        if args.topology_augmentation == "sweep"
        else []
    )
    topology_counts: Counter[str] = Counter()

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        if args.exploration_schedule == "linear-episode-75pct":
            fraction = min(1.0, episode / max(1, round(0.75 * args.episodes) - 1))
            for agent in agents.values():
                agent.episode_epsilon = 0.05 + 0.85 * (1.0 - fraction)
        if trace_data is not None:
            train_tasks = trace_data.train
            if args.contiguous_trace_windows:
                max_offset = max(0, len(train_tasks) - args.tasks)
                offset = random.Random(episode_seed).randint(0, max_offset)
                sample = train_tasks[offset:offset + args.tasks]
            else:
                sample = sample_tasks(train_tasks, args.tasks, episode_seed)
            tasks = compress_arrivals(
                sample,
                args.arrival_compression,
            )
            scenario_name = "alibaba2017_train"
            config = trace_data.simulation_config
            if topology_schedule:
                variant = topology_schedule[episode]
                config = build_topology_variant(config, variant)
                scenario_name = f"alibaba2017_train:{variant.name}"
                topology_counts[variant.name] += 1
            backend = SimPyBackend(config=config)
        else:
            scenario = scenarios[episode % len(scenarios)]
            tasks = compress_arrivals(
                scenario.tasks(args.tasks, episode_seed), args.arrival_compression
            )
            scenario_name = scenario.name
            backend = scenario.backend()
        hier_results, logs_by_tier = run_hierarchical(
            tasks, agents, seed=episode_seed, feedback_mode=args.feedback,
            feedback_frequency=args.feedback_frequency,
            event_threshold=args.event_threshold, training=True,
            backend=backend, freeze_cloud=freeze_cloud,
        )
        feasibility = deadline_feasibility(tasks, backend)
        for tier, log in logs_by_tier.items():
            if tier == "cloud" and freeze_cloud:
                continue
            mean_decision_reward = update_dqn_from_episode(
                agents[tier], log, tasks, hier_results, feasibility,
                credit_mode=args.credit_mode,
                weights=reward_weights,
            )
            for entry in log.diagnostics:
                decision_history.append({
                    "episode": episode + 1,
                    "seed": episode_seed,
                    "tier": tier,
                    **entry,
                    "configuration_hash": config_hash,
                })
            losses = agents[tier].pop_training_losses()
            history.append({
                "episode": episode + 1, "seed": episode_seed, "split": "train",
                "scenario": scenario_name, "tier": tier,
                "decisions": len(log.entries), "action_0": sum(a == 0 for _, _, a in log.entries),
                "action_1": sum(a == 1 for _, _, a in log.entries),
                "action_2": sum(a == 2 for _, _, a in log.entries),
                "mean_decision_reward": mean_decision_reward,
                "team_task_reward": mean_task_reward(
                    tasks, hier_results, feasibility, reward_weights
                ),
                "loss": float(np.mean(losses)) if losses else "",
                "updates": len(losses), "epsilon": agents[tier].epsilon(),
                "deadline_success_rate": float(np.mean([r.deadline_met for r in hier_results])),
                "feasible_task_count": sum(feasibility.values()),
                "feasible_deadline_success_rate": float(np.mean([
                    result.deadline_met for result in hier_results
                    if feasibility[result.task_id]
                ])) if any(feasibility.values()) else 0.0,
                "mean_latency_s": float(np.mean([r.end_to_end_latency_s
                                                   for r in hier_results])),
                "median_latency_s": float(np.median([
                    r.end_to_end_latency_s for r in hier_results
                ])),
                "p95_latency_s": float(np.percentile([
                    r.end_to_end_latency_s for r in hier_results
                ], 95)),
            })

        central_results, central_log = run_centralized(
            tasks, agents["centralized"], seed=episode_seed, training=True,
            backend=backend,
        )
        update_dqn_from_episode(
            agents["centralized"], central_log, tasks, central_results, feasibility,
            credit_mode=args.credit_mode,
            weights=reward_weights,
        )
        for entry in central_log.diagnostics:
            decision_history.append({
                "episode": episode + 1,
                "seed": episode_seed,
                "tier": "centralized",
                **entry,
                "configuration_hash": config_hash,
            })
        losses = agents["centralized"].pop_training_losses()
        history.append({
            "episode": episode + 1, "seed": episode_seed, "split": "train",
            "scenario": scenario_name, "tier": "centralized",
            "decisions": len(central_log.entries),
            "action_0": sum(a == 0 for _, _, a in central_log.entries),
            "action_1": sum(a == 1 for _, _, a in central_log.entries),
            "action_2": sum(a == 2 for _, _, a in central_log.entries),
            "mean_decision_reward": mean_task_reward(
                tasks, central_results, feasibility, reward_weights
            ),
            "team_task_reward": mean_task_reward(
                tasks, central_results, feasibility, reward_weights
            ),
            "loss": float(np.mean(losses)) if losses else "",
            "updates": len(losses), "epsilon": agents["centralized"].epsilon(),
            "deadline_success_rate": float(np.mean([r.deadline_met for r in central_results])),
            "feasible_task_count": sum(feasibility.values()),
            "feasible_deadline_success_rate": float(np.mean([
                result.deadline_met for result in central_results
                if feasibility[result.task_id]
            ])) if any(feasibility.values()) else 0.0,
            "mean_latency_s": float(np.mean([
                r.end_to_end_latency_s for r in central_results])),
            "median_latency_s": float(np.median([
                r.end_to_end_latency_s for r in central_results
            ])),
            "p95_latency_s": float(np.percentile([
                r.end_to_end_latency_s for r in central_results
            ], 95)),
        })
        if (episode + 1) % args.validation_every == 0 or episode + 1 == args.episodes:
            score_hier, score_central, detail = score_validation(
                agents, validation_cases, args.feedback, args.feedback_frequency,
                freeze_cloud=freeze_cloud, weights=reward_weights,
                event_threshold=args.event_threshold,
            )
            hier_key = (score_hier,)
            central_key = (score_central,)
            if args.validation_selection == "deadline-p95":
                hier_key = (detail["hierarchical_deadline"], -detail["hierarchical_p95_latency"])
                central_key = (detail["centralized_deadline"], -detail["centralized_p95_latency"])
            if hier_key > best_keys.get("hierarchical", (float("-inf"),)):
                best_keys["hierarchical"] = hier_key
                best_episodes["hierarchical"] = episode + 1
                best_hier = score_hier
                stale_validation_checks["hierarchical"] = 0
                save_team(
                    agents,
                    args.output,
                    ("device", "edge", "cloud")
                    if args.learn_cloud else ("device", "edge"),
                )
            else:
                stale_validation_checks["hierarchical"] += 1
            if central_key > best_keys.get("centralized", (float("-inf"),)):
                best_keys["centralized"] = central_key
                best_episodes["centralized"] = episode + 1
                best_central = score_central
                stale_validation_checks["centralized"] = 0
                save_team(agents, args.output, ("centralized",))
            else:
                stale_validation_checks["centralized"] += 1
            history.append({
                "episode": episode + 1, "seed": "|".join(map(str, validation_seeds)),
                "split": "validation", "scenario": "all", "tier": "hierarchical",
                "team_task_reward": score_hier,
                "deadline_success_rate": detail["hierarchical_deadline"],
                "feasible_deadline_success_rate": detail["hierarchical_feasible_deadline"],
                "mean_latency_s": detail["hierarchical_latency"],
                "median_latency_s": detail["hierarchical_median_latency"],
                "p95_latency_s": detail["hierarchical_p95_latency"],
            })
            history.append({
                "episode": episode + 1, "seed": "|".join(map(str, validation_seeds)),
                "split": "validation", "scenario": "all", "tier": "centralized",
                "team_task_reward": score_central,
                "deadline_success_rate": detail["centralized_deadline"],
                "feasible_deadline_success_rate": detail["centralized_feasible_deadline"],
                "mean_latency_s": detail["centralized_latency"],
                "median_latency_s": detail["centralized_median_latency"],
                "p95_latency_s": detail["centralized_p95_latency"],
            })
            print(f"Episode {episode + 1}/{args.episodes} | validation reward "
                  f"hier={score_hier:.3f}, central={score_central:.3f} "
                  "(closer to zero is better) | feasible deadline success "
                  f"hier={detail['hierarchical_feasible_deadline']:.3f}, "
                  f"central={detail['centralized_feasible_deadline']:.3f}")
            if all(
                count >= args.patience
                for count in stale_validation_checks.values()
            ):
                print(
                    f"Early stopping after {args.patience} non-improving "
                    "validation checks for both controllers."
                )
                break
        else:
            print(f"Episode {episode + 1}/{args.episodes} | "
                  f"scenario={scenario_name}, completed={len(hier_results)}")

    # Last checkpoint retained independently; root contains the best evaluated set.
    last_members = (
        ("device", "edge", "cloud", "centralized")
        if args.learn_cloud else ("device", "edge", "centralized")
    )
    save_team(agents, args.output / "last", last_members)
    if trace_data is None:
        training_source = "synthetic"
        topology_details = topology_metadata(scenarios[0].backend().config)
        source_details: dict[str, object] = {
            "source": "synthetic",
            "synthetic_scenarios": args.scenarios or [
                "nominal", "high_load", "congested_network"
            ],
        }
    else:
        training_source = trace_data.source
        topology_details = topology_metadata(trace_data.simulation_config)
        source_details = trace_data.metadata
    metadata = {
        "feature_version": FEATURE_VERSION,
        "source": (
            "Synthetic development workload"
            if training_source == "synthetic"
            else "Alibaba Cluster Trace 2017"
        ),
        "workload": source_details,
        "workload_source": training_source,
        "deadline_policy": (
            trace_data.metadata["task_assumptions"]["deadline_policy"]
            if trace_data is not None else "synthetic-scenario"
        ),
        "topology": topology_details,
        "topology_training": {
            "augmentation": args.topology_augmentation,
            "episode_variant_counts": dict(sorted(topology_counts.items())),
            "training_variants": (
                [
                    {
                        "name": variant.name,
                        "description": variant.description,
                        "device_cpu_scale": variant.device_cpu_scale,
                        "edge_cpu_scale": variant.edge_cpu_scale,
                        "cloud_cpu_scale": variant.cloud_cpu_scale,
                        "device_edge_bandwidth_scale": (
                            variant.device_edge_bandwidth_scale
                        ),
                        "edge_cloud_bandwidth_scale": (
                            variant.edge_cloud_bandwidth_scale
                        ),
                        "propagation_scale": variant.propagation_scale,
                        "alternate_mapping": variant.alternate_mapping,
                    }
                    for variant in TRAINING_VARIANTS
                ]
                if topology_schedule
                else []
            ),
            "validation_variants": (
                ["baseline"]
            ),
            "held_out_test_variants": [
                variant.name for variant in HELD_OUT_VARIANTS
            ],
            "deadline_policy": (
                "Deadlines remain derived from the reference topology; route feasibility "
                "is recalculated under each sampled topology."
                if topology_schedule
                else "No topology augmentation."
            ),
            "checkpoint_selection": (
                "Validation reward on the reference topology; four held-out variants "
                "are reserved for final evaluation."
            ),
        },
        "simulator": (
            "SimPyBackend; one execution slot per device/edge and one cloud worker; "
            "queue-wait estimates use active remaining and queued MI"
        ),
        "algorithm": (
            f"{args.algorithm}; same-agent next-decision bootstrap"
            if args.credit_mode == "next-decision"
            else f"{args.algorithm}; delayed terminal credit"
        ),
        "reward_version": REWARD_VERSION,
        "training_seed": args.seed,
        "training_seeds": sorted(training_seeds),
        "validation_seeds": validation_seeds,
        "episodes": args.episodes,
        "episodes_completed": episode + 1,
        "tasks_per_episode": args.tasks,
        "arrival_compression": args.arrival_compression,
        "compression_calibration": "validation" if args.calibrate_on_validation else "configured",
        "reward_weights": {
            "latency": reward_weights.latency,
            "device_energy": reward_weights.device_energy,
            "deadline_miss": reward_weights.deadline_miss,
            "coordination": reward_weights.coordination,
        },
        "feedback_reward_semantics": (
            "The coordination reward uses each task's feedback bytes; feedback modes "
            "therefore optimize different rewards unless coordination weight is zero."
        ),
        "cloud_dispatch_reward": (
            "Negative normalized total queue waiting cost for the selected service "
            "duration; singleton queues are excluded from replay."
        ),
        "credit_mode": args.credit_mode,
        "gamma": gamma,
        "validation_tasks": args.validation_tasks,
        "validation_every": args.validation_every,
        "scenarios": args.scenarios or (
            ["nominal", "high_load", "congested_network"]
            if training_source == "synthetic" else ["alibaba2017"]
        ),
        "actions": {"device": 2, "edge": 2, "cloud": 3, "centralized": 3},
        "cloud_scheduler": "learned-DQN" if args.learn_cloud else "EDF-fixed",
        "dqn": {
            "learning_rate": 3e-4,
            "epsilon_start": 0.90,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 15_000,
            "algorithm": args.algorithm,
            "target_interval": 500,
            "replay_capacity": 8_000,
            "batch_size": args.batch_size,
        },
        "validation_backend_seeds": (
            "One chronological validation split with seeded task-level bandwidth "
            "and service-time noise"
            if trace_data is not None else validation_seeds
        ),
        "feedback": args.feedback,
        "feedback_frequency": args.feedback_frequency,
        "event_threshold": args.event_threshold,
        "stochasticity": {
            "bandwidth_jitter_fraction": (
                trace_data.simulation_config.bandwidth_jitter_fraction
                if trace_data is not None else 0.0
            ),
            "service_time_noise_fraction": (
                trace_data.simulation_config.service_time_noise_fraction
                if trace_data is not None else 0.0
            ),
        },
        "best_validation_reward": {"hierarchical": best_hier, "centralized": best_central},
        "checkpoint_selection": args.validation_selection,
        "selected_episodes": best_episodes,
        "training_window": "contiguous" if args.contiguous_trace_windows else "sampled",
        "exploration_schedule": args.exploration_schedule,
        "centralized_budget_matched": True,
        "early_stopping_patience": args.patience,
        "configuration": experiment_config,
        "configuration_hash": config_hash,
        "note": (
            "Terminal mode is one-step; next-decision mode bootstraps within each "
            "agent's logged decision sequence. Neither mode is a Gymnasium environment."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    with (args.output / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "seed", "split", "scenario", "tier", "decisions", "action_0",
            "action_1", "action_2", "mean_decision_reward", "team_task_reward",
            "loss", "updates", "epsilon", "deadline_success_rate",
            "feasible_task_count", "feasible_deadline_success_rate",
            "mean_latency_s", "median_latency_s", "p95_latency_s",
            "configuration_hash",
        ])
        for row in history:
            row["configuration_hash"] = config_hash
        writer.writeheader()
        writer.writerows(history)
    if decision_history:
        with (args.output / "decision_log.jsonl").open("w", encoding="utf-8") as f:
            for row in decision_history:
                row["observation"] = np.asarray(row["observation"]).tolist()
                q_values = row["q_values"]
                row["q_values"] = None if q_values is None else np.asarray(q_values).tolist()
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print("Saved best-validation checkpoints, last weights and logs to", args.output)


if __name__ == "__main__":
    main()
