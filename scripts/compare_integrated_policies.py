"""Five-policy SimPy comparison plus causal-feedback ablations on held-out seeds.

Development experiments use synthetic tasks and one device/edge/cloud only.
Do NOT cite outputs as evidence of improvements on a public trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mec_offloading.baselines import (
    EDFCloudPolicy, FIFOCloudPolicy, GreedyComputeEdgePolicy,
    GreedyNearestDevicePolicy, RandomCloudPolicy, RandomDevicePolicy,
    RandomEdgePolicy,
)
from mec_offloading.evaluation.runner import _write_csv, aggregate_results, summarize_runs
from mec_offloading.rl.dqn import DQNAgent
from mec_offloading.rl.integrated import run_centralized, run_hierarchical
from mec_offloading.rl.observations import FEATURE_VERSION
from mec_offloading.rl.scenarios import SCENARIOS
from mec_offloading.rl.spaces import SPECS


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


def detailed_rows(label, results, seed, scenario, feedback_mode):
    for result in results:
        yield {
            "policy": label, "scenario": scenario, "feedback_mode": feedback_mode,
            "seed": seed, "task_id": result.task_id,
            "route": "->".join(tier.value for tier in result.route),
            "arrival_time_s": result.arrival_time_s,
            "completion_time_s": result.completion_time_s,
            "execution_start_time_s": result.execution_start_time_s,
            "end_to_end_latency_s": result.end_to_end_latency_s,
            "deadline_met": result.deadline_met,
            "device_energy_j": result.device_energy_j,
            "communication_delay_s": result.communication_delay_s,
            "communication_bytes": result.communication_bytes,
            "feedback_messages": result.feedback_messages,
            "feedback_bytes": result.feedback_bytes,
            "feedback_modeled_delay_s": result.feedback_modeled_delay_s,
            "failure_reason": result.failure_reason or "",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/models/integrated"))
    parser.add_argument("--output", type=Path, default=Path("results/integrated-comparison"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10_001, 10_002, 10_003])
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=["nominal", "high_load", "congested_network"])
    parser.add_argument("--feedback-modes", nargs="+",
                        choices=["none", "periodic", "frequent"],
                        default=["none", "periodic", "frequent"])
    parser.add_argument("--feedback", choices=["none", "periodic", "frequent"],
                        help="Legacy one-mode option; overrides --feedback-modes")
    parser.add_argument("--feedback-frequency", type=int, default=5)
    args = parser.parse_args()
    if args.tasks <= 0 or args.feedback_frequency <= 0:
        parser.error("tasks and feedback frequency must be > 0")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Evaluation seeds must be unique")
    modes = [args.feedback] if args.feedback is not None else list(dict.fromkeys(args.feedback_modes))

    metadata_file = args.models / "metadata.json"
    if not metadata_file.is_file():
        parser.error("No trained model metadata. Run train_integrated_rl.py first")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("feature_version") != FEATURE_VERSION:
        parser.error("Incompatible observation version; retrain integrated agents")
    reserved = set(metadata.get("training_seeds", [])) | set(metadata.get("validation_seeds", []))
    overlap = reserved.intersection(args.seeds)
    if overlap:
        parser.error(f"Data leakage: evaluation seeds overlap train/validation: {sorted(overlap)}")
    agents = {name: DQNAgent.load(args.models / f"{name}.pt")
              for name in ("device", "edge", "cloud", "centralized")}
    for name, agent in agents.items():
        spec = SPECS[name]
        if (agent.obs_dim, agent.n_actions) != (spec.observation_dim, len(spec.actions)):
            parser.error(f"Checkpoint schema/action mismatch for {name}")

    aggregates, details = [], []
    for scenario_name in args.scenarios:
        scenario = SCENARIOS[scenario_name]
        for seed in args.seeds:
            tasks = scenario.tasks(args.tasks, seed)
            experiments = []
            for label, device, edge, cloud in baseline_bundles(seed):
                results = scenario.backend().run_episode(
                    tasks, device, edge, cloud, seed=seed,
                )
                experiments.append((label, results, "n/a"))
            cent_results, _ = run_centralized(
                tasks, agents["centralized"], seed=seed, backend=scenario.backend(),
            )
            experiments.append(("centralized_rl", cent_results, "n/a"))
            for mode in modes:
                hier_results, _ = run_hierarchical(
                    tasks, agents, seed=seed, backend=scenario.backend(),
                    feedback_mode=mode, feedback_frequency=args.feedback_frequency,
                )
                experiments.append((f"hierarchical_{mode}", hier_results, mode))

            for label, results, feedback in experiments:
                row = {"scenario": scenario_name, "feedback_mode": feedback}
                row.update(aggregate_results(label, results, seed=seed))
                aggregates.append(row)
                details.extend(detailed_rows(label, results, seed, scenario_name, feedback))
            print(f"Scenario={scenario_name}, seed={seed}: {len(experiments)} matched policies")

    args.output.mkdir(parents=True, exist_ok=True)
    statistics = []
    for scenario_name in args.scenarios:
        scenario_rows = [row for row in aggregates if row["scenario"] == scenario_name]
        for row in summarize_runs(scenario_rows):
            statistics.append({"scenario": scenario_name, **row})
    _write_csv(args.output / "task_results.csv", details)
    _write_csv(args.output / "aggregate_metrics.csv", aggregates)
    _write_csv(args.output / "aggregate_statistics.csv", statistics)
    (args.output / "metadata.json").write_text(json.dumps({
        "source": "Synthetic development workload (NOT Alibaba/Google trace)",
        "test_seeds": args.seeds,
        "train_seeds": metadata.get("training_seeds", []),
        "validation_seeds": metadata.get("validation_seeds", []),
        "task_count_per_scenario_seed": args.tasks,
        "scenarios": args.scenarios,
        "feedback_modes": modes,
        "feedback_frequency": args.feedback_frequency,
        "training_feature_version": FEATURE_VERSION,
        "simulator": "SimPyBackend, single device/edge/cloud",
        "comparison_policies": [
            "random", "greedy_fifo", "greedy_edf", "centralized_rl",
            *[f"hierarchical_{mode}" for mode in modes],
        ],
        "feedback_note": "Causal decision-time arrival, no control-plane network contention",
        "note": "No claim of real-trace improvement; additional real topology needed",
    }, indent=2), encoding="utf-8")
    print("Comparison and ablations written to", args.output)


if __name__ == "__main__":
    main()
