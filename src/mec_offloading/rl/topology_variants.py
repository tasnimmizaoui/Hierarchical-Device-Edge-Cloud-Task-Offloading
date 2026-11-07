"""Reusable topology perturbations for robust training and sensitivity tests."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from mec_offloading.simulation import SimulationConfig, TopologyConfig


@dataclass(frozen=True)
class TopologyVariant:
    name: str
    description: str
    device_cpu_scale: float = 1.0
    edge_cpu_scale: float = 1.0
    cloud_cpu_scale: float = 1.0
    device_edge_bandwidth_scale: float = 1.0
    edge_cloud_bandwidth_scale: float = 1.0
    propagation_scale: float = 1.0
    alternate_mapping: bool = False


VARIANTS = (
    TopologyVariant("baseline", "Configured reference topology"),
    TopologyVariant(
        "device_cpu_low", "Device CPU capacity at 75% of reference",
        device_cpu_scale=0.75,
    ),
    TopologyVariant(
        "device_cpu_high", "Device CPU capacity at 125% of reference",
        device_cpu_scale=1.25,
    ),
    TopologyVariant(
        "edge_cpu_low", "Edge CPU capacity at 50% of reference",
        edge_cpu_scale=0.5,
    ),
    TopologyVariant(
        "edge_cpu_high", "Edge CPU capacity at 150% of reference",
        edge_cpu_scale=1.5,
    ),
    TopologyVariant(
        "cloud_cpu_low", "Cloud CPU capacity at 50% of reference",
        cloud_cpu_scale=0.5,
    ),
    TopologyVariant(
        "cloud_cpu_high", "Cloud CPU capacity at 150% of reference",
        cloud_cpu_scale=1.5,
    ),
    TopologyVariant(
        "device_edge_link_slow", "Device-edge bandwidth at 10%; propagation doubled",
        device_edge_bandwidth_scale=0.1,
        propagation_scale=2.0,
    ),
    TopologyVariant(
        "device_edge_link_fast", "Device-edge bandwidth doubled; propagation halved",
        device_edge_bandwidth_scale=2.0,
        propagation_scale=0.5,
    ),
    TopologyVariant(
        "edge_cloud_link_slow", "Edge-cloud bandwidth at 10%; propagation doubled",
        edge_cloud_bandwidth_scale=0.1,
        propagation_scale=2.0,
    ),
    TopologyVariant(
        "edge_cloud_link_fast", "Edge-cloud bandwidth doubled; propagation halved",
        edge_cloud_bandwidth_scale=2.0,
        propagation_scale=0.5,
    ),
    TopologyVariant(
        "alternate_edge_mapping", "Swap device-2 and device-3 edge associations",
        alternate_mapping=True,
    ),
    TopologyVariant(
        "combined_stress", "Compute at 75%, links at 50%, propagation doubled",
        device_cpu_scale=0.75,
        edge_cpu_scale=0.75,
        cloud_cpu_scale=0.75,
        device_edge_bandwidth_scale=0.5,
        edge_cloud_bandwidth_scale=0.5,
        propagation_scale=2.0,
    ),
)

VALIDATION_VARIANTS = tuple(
    next(variant for variant in VARIANTS if variant.name == name)
    for name in (
        "baseline",
        "edge_cpu_low",
        "cloud_cpu_low",
        "device_edge_link_slow",
        "edge_cloud_link_slow",
        "alternate_edge_mapping",
        "combined_stress",
    )
)

TRAINING_VARIANTS = tuple(
    next(variant for variant in VARIANTS if variant.name == name)
    for name in (
        "baseline",
        "device_cpu_low",
        "device_cpu_high",
        "edge_cpu_low",
        "edge_cpu_high",
        "cloud_cpu_low",
        "device_edge_link_slow",
        "edge_cloud_link_slow",
        "alternate_edge_mapping",
    )
)

HELD_OUT_VARIANTS = tuple(
    next(variant for variant in VARIANTS if variant.name == name)
    for name in (
        "cloud_cpu_high",
        "device_edge_link_fast",
        "edge_cloud_link_fast",
        "combined_stress",
    )
)


def build_topology_variant(
    config: SimulationConfig,
    variant: TopologyVariant,
) -> SimulationConfig:
    """Apply a named sensitivity change without modifying the reference config."""
    topology = config.topology
    if topology is None:
        raise ValueError("Topology variants require an explicit device/edge topology")
    devices = tuple(
        replace(device, cpu_capacity_mips=device.cpu_capacity_mips * variant.device_cpu_scale)
        for device in topology.devices
    )
    edges = tuple(
        replace(
            edge,
            cpu_capacity_mips=edge.cpu_capacity_mips * variant.edge_cpu_scale,
            device_edge_bandwidth_mbps=(
                edge.device_edge_bandwidth_mbps * variant.device_edge_bandwidth_scale
            ),
            edge_cloud_bandwidth_mbps=(
                edge.edge_cloud_bandwidth_mbps * variant.edge_cloud_bandwidth_scale
            ),
            device_edge_propagation_s=(
                edge.device_edge_propagation_s * variant.propagation_scale
            ),
            edge_cloud_propagation_s=(
                edge.edge_cloud_propagation_s * variant.propagation_scale
            ),
        )
        for edge in topology.edges
    )
    associations = dict(topology.device_to_edge)
    if variant.alternate_mapping:
        if {"device-2", "device-3"} - associations.keys():
            raise ValueError("Alternate mapping requires device-2 and device-3")
        associations["device-2"], associations["device-3"] = (
            associations["device-3"],
            associations["device-2"],
        )
    return replace(
        config,
        cloud_cpu_capacity_mips=config.cloud_cpu_capacity_mips * variant.cloud_cpu_scale,
        topology=TopologyConfig(devices, edges, tuple(associations.items())),
    )


def shuffled_variant_schedule(
    variants: tuple[TopologyVariant, ...],
    episode_count: int,
    seed: int,
) -> list[TopologyVariant]:
    """Return a seeded, near-balanced schedule; each cycle includes every variant."""
    if not variants:
        raise ValueError("At least one topology variant is required")
    if episode_count <= 0:
        raise ValueError("episode_count must be > 0")
    rng = random.Random(seed)
    schedule: list[TopologyVariant] = []
    while len(schedule) < episode_count:
        cycle = list(variants)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:episode_count]
