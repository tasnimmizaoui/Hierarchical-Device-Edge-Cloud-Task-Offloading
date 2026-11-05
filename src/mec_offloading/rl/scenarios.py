"""Development-only synthetic load/network variants (NOT public traces).

Only creates input tasks and SimulationConfig instances; the Student 1
simulation backend, workload module and baseline implementations stay intact.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from mec_offloading.domain.models import Task
from mec_offloading.simulation.simpy_backend import SimPyBackend, SimulationConfig
from mec_offloading.workloads import generate_synthetic_tasks


@dataclass(frozen=True)
class DevelopmentScenario:
    name: str
    interarrival_scale: float = 1.0
    bandwidth_scale: float = 1.0
    propagation_scale: float = 1.0

    def __post_init__(self) -> None:
        if min(self.interarrival_scale, self.bandwidth_scale,
               self.propagation_scale) <= 0:
            raise ValueError("Scenario scales must be positive")

    def tasks(self, count: int, seed: int) -> list[Task]:
        return [replace(t, arrival_time_s=t.arrival_time_s * self.interarrival_scale)
                for t in generate_synthetic_tasks(count=count, seed=seed)]

    def backend(self) -> SimPyBackend:
        base = SimulationConfig()
        return SimPyBackend(config=replace(
            base,
            device_edge_bandwidth_mbps=(base.device_edge_bandwidth_mbps
                                        * self.bandwidth_scale),
            edge_cloud_bandwidth_mbps=(base.edge_cloud_bandwidth_mbps
                                       * self.bandwidth_scale),
            device_edge_propagation_s=(base.device_edge_propagation_s
                                       * self.propagation_scale),
            edge_cloud_propagation_s=(base.edge_cloud_propagation_s
                                      * self.propagation_scale),
        ))


SCENARIOS = {
    "nominal": DevelopmentScenario("nominal"),
    "low_load": DevelopmentScenario("low_load", interarrival_scale=2.0),
    "high_load": DevelopmentScenario("high_load", interarrival_scale=0.5),
    "congested_network": DevelopmentScenario(
        "congested_network", bandwidth_scale=0.25, propagation_scale=3.0,
    ),
}
