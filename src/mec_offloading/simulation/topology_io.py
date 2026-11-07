"""Load shared topology configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from mec_offloading.domain.models import DeviceState
from mec_offloading.simulation.simpy_backend import (
    EdgeConfig,
    SimulationConfig,
    TopologyConfig,
)


def load_topology_config(path: str | Path) -> SimulationConfig:
    """Build the simulator configuration from the project's YAML topology shape."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    topology_data = data["topology"]
    devices = tuple(
        DeviceState(
            device["id"],
            device["cpu_mips"],
            device["battery_j"],
            device["battery_capacity_j"],
            device["tx_power_w"],
            device["rx_power_w"],
            device["cpu_energy_per_mi_j"],
        )
        for device in topology_data["devices"]
    )
    edges = tuple(
        EdgeConfig(
            edge["id"],
            edge["cpu_mips"],
            edge["device_edge_bandwidth_mbps"],
            edge["edge_cloud_bandwidth_mbps"],
            edge["device_edge_propagation_s"],
            edge["edge_cloud_propagation_s"],
        )
        for edge in topology_data["edges"]
    )
    topology = TopologyConfig(
        devices,
        edges,
        tuple(topology_data["device_to_edge"].items()),
    )
    return SimulationConfig(
        cloud_cpu_capacity_mips=data["resources"]["cloud_cpu_mips"],
        bandwidth_jitter_fraction=data["resources"].get(
            "bandwidth_jitter_fraction", 0.0
        ),
        service_time_noise_fraction=data["resources"].get(
            "service_time_noise_fraction", 0.0
        ),
        topology=topology,
    )
