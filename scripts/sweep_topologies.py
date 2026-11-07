"""Measure held-out policy sensitivity to compute, links, and edge placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mec_offloading.baselines import (
    EDFCloudPolicy,
    GreedyComputeEdgePolicy,
    GreedyNearestDevicePolicy,
)
from mec_offloading.domain.models import Task
from mec_offloading.evaluation.runner import _write_csv, aggregate_results
from mec_offloading.rl.credit import REWARD_VERSION, deadline_feasibility
from mec_offloading.rl.dqn import DQNAgent
from mec_offloading.rl.experiment_data import (
    DEFAULT_TRACE_ARRIVAL_COMPRESSION,
    compress_arrivals,
    load_trace_experiment,
    topology_metadata,
)
from mec_offloading.rl.integrated import run_centralized, run_hierarchical
from mec_offloading.rl.observations import FEATURE_VERSION
from mec_offloading.rl.spaces import SPECS
from mec_offloading.rl.topology_variants import (
    HELD_OUT_VARIANTS,
    TopologyVariant,
    build_topology_variant,
)
from mec_offloading.simulation import SimPyBackend, SimulationConfig
from mec_offloading.trace_workloads import ALIBABA_2017_ADAPTER_VERSION


def _validate_checkpoint_metadata(metadata: dict[str, object], data) -> None:
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint observation features differ; retrain the models")
    if metadata.get("reward_version") != REWARD_VERSION:
        raise ValueError("Checkpoint reward version differs; retrain the models")
    if metadata.get("workload_source") != "alibaba2017":
        raise ValueError("Topology sweep expects Alibaba 2017 checkpoints")
    workload = metadata.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("Checkpoint is missing workload metadata")
    if workload.get("trace_adapter_version") != ALIBABA_2017_ADAPTER_VERSION:
        raise ValueError("Checkpoint uses an incompatible Alibaba task adapter")
    if metadata.get("deadline_policy") != data.metadata["task_assumptions"]["deadline_policy"]:
        raise ValueError("Checkpoint deadline policy differs from the requested evaluation")
    if metadata.get("topology") != topology_metadata(data.simulation_config):
        raise ValueError("Checkpoint was not trained on the reference topology")


def run_sweep(
    tasks: list[Task],
    config: SimulationConfig,
    agents: dict[str, DQNAgent],
    *,
    seed: int,
    variants: tuple[TopologyVariant, ...] = HELD_OUT_VARIANTS,
) -> list[dict[str, object]]:
    """Evaluate greedy, centralized, and hierarchical policies per topology."""
    rows: list[dict[str, object]] = []
    for variant in variants:
        variant_config = build_topology_variant(config, variant)
        backend = SimPyBackend(config=variant_config)
        feasible = deadline_feasibility(tasks, backend)
        policy_results = {
            "greedy_edf": backend.run_episode(
                tasks,
                GreedyNearestDevicePolicy(),
                GreedyComputeEdgePolicy(),
                EDFCloudPolicy(),
                seed=seed,
            ),
            "centralized_rl": run_centralized(
                tasks,
                agents["centralized"],
                seed=seed,
                backend=backend,
            )[0],
        }
        for mode in ("none", "periodic", "frequent"):
            policy_results[f"hierarchical_{mode}"] = run_hierarchical(
                tasks,
                agents,
                seed=seed,
                feedback_mode=mode,
                freeze_cloud=True,
                backend=backend,
            )[0]
        for policy, results in policy_results.items():
            row = aggregate_results(policy, results, seed=seed)
            feasible_results = [result for result in results if feasible[result.task_id]]
            row.update({
                "variant": variant.name,
                "variant_description": variant.description,
                "feasible_task_count": len(feasible_results),
                "infeasible_task_count": len(results) - len(feasible_results),
                "feasible_deadline_success_rate": (
                    sum(result.deadline_met for result in feasible_results)
                    / len(feasible_results)
                    if feasible_results
                    else 0.0
                ),
            })
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("artifacts/models/alibaba-best-tier-calibrated-v8"),
    )
    parser.add_argument(
        "--topology-config",
        type=Path,
        default=Path("configs/alibaba-multi-topology.yaml"),
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("data/raw/batch_task.csv"),
    )
    parser.add_argument(
        "--trace-instances",
        type=Path,
        default=Path("data/raw/batch_instance.csv"),
    )
    parser.add_argument(
        "--deadline-policy",
        choices=["fixed-5s", "local-relative-2x", "best-tier-calibrated"],
        default="best-tier-calibrated",
    )
    parser.add_argument("--max-trace-tasks", type=int, default=30_000)
    parser.add_argument("--tasks", type=int, default=579)
    parser.add_argument("--arrival-compression", type=float)
    parser.add_argument("--seed", type=int, default=10_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/alibaba-topology-sensitivity-v8"),
    )
    args = parser.parse_args()
    if args.tasks <= 0 or args.max_trace_tasks < 3:
        parser.error("--tasks must be > 0 and --max-trace-tasks must be >= 3")

    try:
        data = load_trace_experiment(
            args.trace_path,
            args.trace_instances,
            args.topology_config,
            max_tasks=args.max_trace_tasks,
            deadline_policy=args.deadline_policy,
        )
        metadata_path = args.models / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_checkpoint_metadata(metadata, data)
        arrival_compression = (
            metadata.get("arrival_compression", DEFAULT_TRACE_ARRIVAL_COMPRESSION)
            if args.arrival_compression is None
            else args.arrival_compression
        )
        if arrival_compression <= 0:
            raise ValueError("arrival compression must be > 0")
        if metadata.get("arrival_compression", 1.0) != arrival_compression:
            raise ValueError("Arrival compression differs from the training checkpoint")
        tasks = compress_arrivals(list(data.test[: args.tasks]), arrival_compression)
        if args.tasks < len(data.test):
            raise ValueError(
                "For a reproducible held-out sweep, --tasks must cover the full test split "
                f"({len(data.test)} tasks)"
            )
        agents = {
            name: DQNAgent.load(args.models / f"{name}.pt")
            for name in ("device", "edge", "centralized")
        }
        for name, agent in agents.items():
            spec = SPECS[name]
            if (agent.obs_dim, agent.n_actions) != (
                spec.observation_dim,
                len(spec.actions),
            ):
                raise ValueError(f"{name} checkpoint schema/action mismatch")
        rows = run_sweep(
            tasks,
            data.simulation_config,
            agents,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Could not run topology sensitivity sweep: {error}")

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "topology_metrics.csv", rows)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "source": "Alibaba Cluster Trace 2017",
                "deadline_policy": args.deadline_policy,
                "topology_reference": str(args.topology_config),
                "reference_topology": topology_metadata(data.simulation_config),
                "checkpoint_directory": str(args.models),
                "test_task_count": len(tasks),
                "seed": args.seed,
                "arrival_compression": arrival_compression,
                "topology_variants": [
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
                    for variant in HELD_OUT_VARIANTS
                ],
                "deadline_interpretation": (
                    "Deadlines remain fixed from the selected baseline policy; "
                    "feasibility is recalculated per variant."
                ),
                "simulator": (
                    "One execution slot per device/edge and one cloud worker; "
                    "no shared network contention."
                ),
                "note": (
                    "Sensitivity analysis only; variant capacities and links are "
                    "hypothetical, not measured deployments."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} policy-variant rows to {args.output}")


if __name__ == "__main__":
    main()
