from __future__ import annotations

from pathlib import Path

import pytest

from mec_offloading.rl.topology_variants import (
    HELD_OUT_VARIANTS,
    TRAINING_VARIANTS,
    VARIANTS,
    build_topology_variant,
    shuffled_variant_schedule,
)
from mec_offloading.simulation import load_topology_config


@pytest.fixture
def reference_config():
    root = Path(__file__).resolve().parents[1]
    return load_topology_config(root / "configs" / "alibaba-multi-topology.yaml")


def test_sweep_has_reference_and_one_factor_variants():
    names = {variant.name for variant in VARIANTS}

    assert "baseline" in names
    assert "device_cpu_low" in names
    assert "edge_cpu_low" in names
    assert "cloud_cpu_low" in names
    assert "device_edge_link_slow" in names
    assert "edge_cloud_link_slow" in names
    assert "alternate_edge_mapping" in names
    assert len(TRAINING_VARIANTS) == 9
    assert len(HELD_OUT_VARIANTS) == 4
    assert {variant.name for variant in TRAINING_VARIANTS}.isdisjoint(
        variant.name for variant in HELD_OUT_VARIANTS
    )
    assert "combined_stress" in {variant.name for variant in HELD_OUT_VARIANTS}


def test_reference_trace_topology_scales_devices_and_enables_seeded_noise(reference_config):
    topology = reference_config.topology
    assert topology is not None
    assert len(topology.devices) == 24
    assert len(topology.edges) == 3
    assert [device.cpu_capacity_mips for device in topology.devices[:3]] == [
        800.0, 1000.0, 1200.0
    ]
    assert reference_config.bandwidth_jitter_fraction == .1
    assert reference_config.service_time_noise_fraction == .1


def test_cpu_and_link_variants_change_only_requested_values(reference_config):
    baseline = reference_config.topology
    assert baseline is not None
    variant = next(item for item in VARIANTS if item.name == "edge_cpu_low")

    changed = build_topology_variant(reference_config, variant)

    assert changed.topology is not None
    assert [
        edge.cpu_capacity_mips for edge in changed.topology.edges
    ] == [edge.cpu_capacity_mips * 0.5 for edge in baseline.edges]
    assert changed.cloud_cpu_capacity_mips == reference_config.cloud_cpu_capacity_mips
    assert [
        device.cpu_capacity_mips for device in changed.topology.devices
    ] == [device.cpu_capacity_mips for device in baseline.devices]
    assert changed.topology.device_to_edge == baseline.device_to_edge


def test_alternate_mapping_swaps_only_the_two_target_devices(reference_config):
    topology = reference_config.topology
    assert topology is not None
    variant = next(item for item in VARIANTS if item.name == "alternate_edge_mapping")

    changed = build_topology_variant(reference_config, variant)

    assert changed.topology is not None
    original = dict(topology.device_to_edge)
    swapped = dict(changed.topology.device_to_edge)
    assert swapped["device-2"] == original["device-3"]
    assert swapped["device-3"] == original["device-2"]
    assert {
        device_id: edge_id
        for device_id, edge_id in swapped.items()
        if device_id not in {"device-2", "device-3"}
    } == {
        device_id: edge_id
        for device_id, edge_id in original.items()
        if device_id not in {"device-2", "device-3"}
    }


def test_training_schedule_is_seeded_and_near_balanced():
    first = shuffled_variant_schedule(VARIANTS, 30, 13)
    second = shuffled_variant_schedule(VARIANTS, 30, 13)

    assert first == second
    counts = {variant.name: sum(item == variant for item in first) for variant in VARIANTS}
    assert max(counts.values()) - min(counts.values()) <= 1
    assert set(counts.values()) == {2, 3}


def test_training_schedule_rejects_empty_variants_and_nonpositive_episodes():
    with pytest.raises(ValueError, match="At least one"):
        shuffled_variant_schedule((), 3, 1)
    with pytest.raises(ValueError, match="episode_count"):
        shuffled_variant_schedule(VARIANTS, 0, 1)
