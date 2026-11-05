"""Train independent Double-DQN agents on the *canonical SimPy* backend.

Development-only synthetic data: separate train/validation seeds, repeatable
load/network scenarios, per-decision terminal credit, best-validation checkpoints.
No public trace or real multi-device topology is claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch

from mec_offloading.rl.integrated import (
    make_agents, run_centralized, run_hierarchical, team_reward,
    update_dqn_from_episode,
)
from mec_offloading.rl.observations import FEATURE_VERSION
from mec_offloading.rl.scenarios import SCENARIOS


def mean_task_reward(tasks, results) -> float:
    outcomes = {r.task_id: r for r in results}
    if len(outcomes) != len(tasks):
        raise RuntimeError("Missing completed tasks in SimPy run")
    return float(np.mean([team_reward(task, outcomes[task.task_id])
                          for task in tasks]))


def score_validation(agents, scenarios, seeds: list[int], count: int,
                     feedback: str, frequency: int) -> tuple[float, float, dict]:
    """Evaluate greedily, without replay updates or exploration."""
    hierarchical, centralized = [], []
    metrics = {"hierarchical_deadline": [], "centralized_deadline": [],
               "hierarchical_latency": [], "centralized_latency": []}
    for scenario in scenarios:
        for seed in seeds:
            tasks = scenario.tasks(count, seed)
            hier, _ = run_hierarchical(
                tasks, agents, seed=seed, backend=scenario.backend(),
                feedback_mode=feedback, feedback_frequency=frequency,
                training=False,
            )
            cent, _ = run_centralized(
                tasks, agents["centralized"], seed=seed,
                backend=scenario.backend(), training=False,
            )
            hierarchical.append(mean_task_reward(tasks, hier))
            centralized.append(mean_task_reward(tasks, cent))
            for label, results in (("hierarchical", hier), ("centralized", cent)):
                metrics[f"{label}_deadline"].append(np.mean([r.deadline_met for r in results]))
                metrics[f"{label}_latency"].append(np.mean([
                    r.end_to_end_latency_s for r in results]))
    return float(np.mean(hierarchical)), float(np.mean(centralized)), {
        key: float(np.mean(values)) for key, values in metrics.items()
    }


def save_team(agents, output: Path, members: tuple[str, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in members:
        agents[name].save(output / f"{name}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feedback", choices=["none", "periodic", "frequent"],
                        default="periodic")
    parser.add_argument("--feedback-frequency", type=int, default=5)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=["nominal", "high_load", "congested_network"])
    parser.add_argument("--validation-every", type=int, default=3)
    parser.add_argument("--validation-tasks", type=int, default=40)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/integrated"))
    args = parser.parse_args()
    if min(args.episodes, args.tasks, args.feedback_frequency, args.batch_size,
           args.validation_every, args.validation_tasks) <= 0:
        parser.error("All episode, task, batch, frequency and validation counts must be > 0")
    scenarios = [SCENARIOS[name] for name in args.scenarios]
    validation_seeds = (args.validation_seeds if args.validation_seeds is not None
                        else [args.seed + 100_000, args.seed + 100_001])
    training_seeds = {args.seed + i for i in range(args.episodes)}
    if training_seeds.intersection(validation_seeds):
        parser.error("Validation seeds must be disjoint from all training seeds")
    if len(set(validation_seeds)) != len(validation_seeds):
        parser.error("Validation seeds must be unique")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    agents = make_agents(args.seed, batch_size=args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_hier = float("-inf")
    best_central = float("-inf")

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        scenario = scenarios[episode % len(scenarios)]
        tasks = scenario.tasks(args.tasks, episode_seed)
        hier_results, logs_by_tier = run_hierarchical(
            tasks, agents, seed=episode_seed, feedback_mode=args.feedback,
            feedback_frequency=args.feedback_frequency, training=True,
            backend=scenario.backend(),
        )
        for tier, log in logs_by_tier.items():
            mean_decision_reward = update_dqn_from_episode(
                agents[tier], log, tasks, hier_results,
            )
            losses = agents[tier].pop_training_losses()
            history.append({
                "episode": episode + 1, "seed": episode_seed, "split": "train",
                "scenario": scenario.name, "tier": tier,
                "decisions": len(log.entries), "action_0": sum(a == 0 for _, _, a in log.entries),
                "action_1": sum(a == 1 for _, _, a in log.entries),
                "action_2": sum(a == 2 for _, _, a in log.entries),
                "mean_decision_reward": mean_decision_reward,
                "team_task_reward": mean_task_reward(tasks, hier_results),
                "loss": float(np.mean(losses)) if losses else "",
                "updates": len(losses), "epsilon": agents[tier].epsilon(),
                "deadline_success_rate": float(np.mean([r.deadline_met for r in hier_results])),
                "mean_latency_s": float(np.mean([r.end_to_end_latency_s
                                                   for r in hier_results])),
            })

        central_results, central_log = run_centralized(
            tasks, agents["centralized"], seed=episode_seed, training=True,
            backend=scenario.backend(),
        )
        update_dqn_from_episode(agents["centralized"], central_log, tasks, central_results)
        losses = agents["centralized"].pop_training_losses()
        history.append({
            "episode": episode + 1, "seed": episode_seed, "split": "train",
            "scenario": scenario.name, "tier": "centralized",
            "decisions": len(central_log.entries),
            "action_0": sum(a == 0 for _, _, a in central_log.entries),
            "action_1": sum(a == 1 for _, _, a in central_log.entries),
            "action_2": sum(a == 2 for _, _, a in central_log.entries),
            "mean_decision_reward": mean_task_reward(tasks, central_results),
            "team_task_reward": mean_task_reward(tasks, central_results),
            "loss": float(np.mean(losses)) if losses else "",
            "updates": len(losses), "epsilon": agents["centralized"].epsilon(),
            "deadline_success_rate": float(np.mean([r.deadline_met for r in central_results])),
            "mean_latency_s": float(np.mean([
                r.end_to_end_latency_s for r in central_results])),
        })
        if (episode + 1) % args.validation_every == 0 or episode + 1 == args.episodes:
            score_hier, score_central, detail = score_validation(
                agents, scenarios, validation_seeds, args.validation_tasks,
                args.feedback, args.feedback_frequency,
            )
            if score_hier > best_hier:
                best_hier = score_hier
                save_team(agents, args.output, ("device", "edge", "cloud"))
            if score_central > best_central:
                best_central = score_central
                save_team(agents, args.output, ("centralized",))
            history.append({
                "episode": episode + 1, "seed": "|".join(map(str, validation_seeds)),
                "split": "validation", "scenario": "all", "tier": "hierarchical",
                "team_task_reward": score_hier,
                "deadline_success_rate": detail["hierarchical_deadline"],
                "mean_latency_s": detail["hierarchical_latency"],
            })
            history.append({
                "episode": episode + 1, "seed": "|".join(map(str, validation_seeds)),
                "split": "validation", "scenario": "all", "tier": "centralized",
                "team_task_reward": score_central,
                "deadline_success_rate": detail["centralized_deadline"],
                "mean_latency_s": detail["centralized_latency"],
            })
            print(f"Episode {episode + 1}/{args.episodes} | validation reward "
                  f"hier={score_hier:.3f}, central={score_central:.3f} "
                  "(larger is better)")
        else:
            print(f"Episode {episode + 1}/{args.episodes} | "
                  f"scenario={scenario.name}, completed={len(hier_results)}")

    # Last checkpoint retained independently; root contains the best evaluated set.
    save_team(agents, args.output / "last", tuple(agents))
    metadata = {
        "feature_version": FEATURE_VERSION,
        "source": "synthetic task generator; NO Alibaba/Google trace loaded",
        "simulator": "SimPyBackend: one device / one edge / one cloud",
        "algorithm": "Double-DQN; delayed *terminal per-task* credit (one-step formulation)",
        "training_seed": args.seed,
        "training_seeds": sorted(training_seeds),
        "validation_seeds": validation_seeds,
        "episodes": args.episodes,
        "tasks_per_episode": args.tasks,
        "validation_tasks": args.validation_tasks,
        "validation_every": args.validation_every,
        "scenarios": args.scenarios,
        "actions": {"device": 2, "edge": 2, "cloud": 3, "centralized": 3},
        "feedback": args.feedback,
        "feedback_frequency": args.feedback_frequency,
        "best_validation_reward": {"hierarchical": best_hier, "centralized": best_central},
        "checkpoint_selection": "Best validation reward; last/ contains final weights",
        "note": "Development training only; final scientific benchmark pending Student 1 trace/topology",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    with (args.output / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "seed", "split", "scenario", "tier", "decisions", "action_0",
            "action_1", "action_2", "mean_decision_reward", "team_task_reward",
            "loss", "updates", "epsilon", "deadline_success_rate", "mean_latency_s",
        ])
        writer.writeheader()
        writer.writerows(history)
    print("Saved best-validation checkpoints, last weights and logs to", args.output)


if __name__ == "__main__":
    main()
