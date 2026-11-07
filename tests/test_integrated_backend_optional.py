"""End-to-end integration tests (run when SimPy is installed)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("simpy")

from mec_offloading.rl.integrated import run_centralized, run_hierarchical, team_reward
from mec_offloading.simulation import SimPyBackend, load_topology_config
from mec_offloading.trace_workloads import assign_tasks_to_devices
from mec_offloading.workloads import generate_synthetic_tasks


class FixedAgent:
    def __init__(self, action: int):
        self.action = action

    def act(self, observation, training=False):
        return self.action


def test_rl_all_cloud_and_coordination_accounting():
    tasks = generate_synthetic_tasks(5, seed=11)
    agents = {"device": FixedAgent(1), "edge": FixedAgent(1), "cloud": FixedAgent(0)}
    outcomes, logs = run_hierarchical(tasks, agents, seed=11, feedback_mode="frequent")
    assert len(outcomes) == 5
    assert all(result.route[-1].value == "cloud" for result in outcomes)
    assert all(result.feedback_messages == 2 for result in outcomes)
    assert all(result.feedback_bytes == 64 for result in outcomes)
    assert len(logs["device"].entries) == 5
    assert len(logs["edge"].entries) == 5
    assert len(logs["cloud"].entries) <= 5
    assert all(
        team_reward(task, outcome) <= 0
        for task, outcome in zip(tasks, outcomes, strict=True)
    )


def test_hierarchical_frozen_cloud_uses_edf_without_cloud_agent():
    tasks = generate_synthetic_tasks(3, seed=14)
    agents = {"device": FixedAgent(1), "edge": FixedAgent(1)}

    outcomes, logs = run_hierarchical(
        tasks,
        agents,
        seed=14,
        freeze_cloud=True,
    )

    assert len(outcomes) == len(tasks)
    assert all(result.route[-1].value == "cloud" for result in outcomes)
    assert logs["device"].entries
    assert logs["edge"].entries
    assert logs["cloud"].entries == []


def test_centralized_cloud_routing_and_global_observation():
    tasks = generate_synthetic_tasks(4, seed=12)
    outcomes, logs = run_centralized(tasks, FixedAgent(2), seed=12)
    assert len(outcomes) == len(tasks)
    assert len(logs.entries) == len(tasks)
    assert all(result.route[-1].value == "cloud" for result in outcomes)
    assert all(result.feedback_bytes == 0 for result in outcomes)


def test_hierarchical_rl_runs_across_configured_devices_and_edges():
    root = Path(__file__).resolve().parents[1]
    config = load_topology_config(root / "configs" / "alibaba-multi-topology.yaml")
    assert config.topology is not None
    device_ids = tuple(device.device_id for device in config.topology.devices)
    tasks = assign_tasks_to_devices(
        generate_synthetic_tasks(len(device_ids), seed=19),
        device_ids,
    )
    agents = {"device": FixedAgent(1), "edge": FixedAgent(1), "cloud": FixedAgent(0)}

    outcomes, logs = run_hierarchical(
        tasks,
        agents,
        seed=19,
        feedback_mode="frequent",
        backend=SimPyBackend(config=config),
    )

    assert {result.source_device_id for result in outcomes} == set(device_ids)
    assert {result.source_edge_id for result in outcomes} == {
        "edge-1", "edge-2", "edge-3"
    }
    assert all(result.route[-1].value == "cloud" for result in outcomes)
    assert all(result.feedback_messages == 2 for result in outcomes)
    assert len(logs["device"].entries) == len(tasks)
    assert len(logs["edge"].entries) == len(tasks)
