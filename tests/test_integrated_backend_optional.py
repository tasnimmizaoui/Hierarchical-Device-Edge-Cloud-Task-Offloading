"""End-to-end integration tests (run when SimPy is installed)."""
from __future__ import annotations

import pytest

pytest.importorskip("simpy")

from mec_offloading.rl.integrated import run_centralized, run_hierarchical, team_reward
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
    assert len(logs["cloud"].entries) == 5
    assert all(team_reward(task, outcome) <= 0 for task, outcome in zip(tasks, outcomes))


def test_centralized_cloud_routing_and_global_observation():
    tasks = generate_synthetic_tasks(4, seed=12)
    outcomes, logs = run_centralized(tasks, FixedAgent(2), seed=12)
    assert len(outcomes) == len(tasks)
    assert len(logs.entries) == len(tasks)
    assert all(result.route[-1].value == "cloud" for result in outcomes)
    assert all(result.feedback_bytes == 0 for result in outcomes)
