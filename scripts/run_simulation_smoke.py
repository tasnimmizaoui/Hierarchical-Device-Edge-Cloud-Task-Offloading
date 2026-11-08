"""Run the three mandatory reference routes and print measured task outcomes."""

from __future__ import annotations

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    LocalOnlyPolicy,
)
from mec_offloading.simulation import SimPyBackend
from mec_offloading.workloads import generate_synthetic_tasks


def main() -> None:
    backend = SimPyBackend()
    tasks = generate_synthetic_tasks(count=100, seed=1001)
    scenarios = (
        ("local", LocalOnlyPolicy(), EdgeExecutePolicy()),
        ("edge", AlwaysEdgePolicy(), EdgeExecutePolicy()),
        ("cloud", AlwaysEdgePolicy(), EdgeForwardPolicy()),
    )
    for name, device_policy, edge_policy in scenarios:
        results = backend.run_episode(
            tasks,
            device_policy,
            edge_policy,
            EDFCloudPolicy(),
            seed=7,
        )
        print(name)
        for result in results:
            print(
                f"  {result.task_id}: route={'->'.join(tier.value for tier in result.route)}, "
                f"latency={result.end_to_end_latency_s:.4f}s, "
                f"energy={result.device_energy_j:.4f}J, "
                f"deadline_met={result.deadline_met}, "
                f"bytes={result.communication_bytes}"
            )


if __name__ == "__main__":
    main()
