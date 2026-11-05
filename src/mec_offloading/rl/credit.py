"""Per-task terminal credit and explicit normalized multi-objective reward.

This module deliberately has no SimPy dependency, so reward and credit
semantics can be unit-tested while the simulator integration is pending.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mec_offloading.domain.models import Task, TaskResult
from mec_offloading.rl.dqn import DQNAgent, Transition
from mec_offloading.rl.policies import DecisionLog

@dataclass(frozen=True)
class RewardWeights:
    latency: float = 0.35
    device_energy: float = 0.25
    deadline_miss: float = 0.35
    coordination: float = 0.05
    energy_reference_j: float = 10.0
    communication_reference_bytes: float = 64.0


def team_reward(task: Task, result: TaskResult,
                weights: RewardWeights = RewardWeights()) -> float:
    """Shared per-task terminal reward; bounded dimensionless penalty."""
    if weights.energy_reference_j <= 0 or weights.communication_reference_bytes <= 0:
        raise ValueError("Reward normalization references must be positive")
    if min(weights.latency, weights.device_energy, weights.deadline_miss,
           weights.coordination) < 0:
        raise ValueError("Reward weights must be nonnegative")
    latency_fraction = min(3.0, result.end_to_end_latency_s / task.deadline_s)
    energy_fraction = min(3.0, result.device_energy_j / weights.energy_reference_j)
    communication_fraction = min(3.0, result.feedback_bytes /
                                 weights.communication_reference_bytes)
    return -float(weights.latency * latency_fraction
                  + weights.device_energy * energy_fraction
                  + weights.deadline_miss * (not result.deadline_met)
                  + weights.coordination * communication_fraction)


def update_dqn_from_episode(agent: DQNAgent, log: DecisionLog,
                            tasks: list[Task], results: list[TaskResult]) -> float:
    """Assign terminal credit to decisions only after the full episode ends.

    Every (observation, action) belongs to its own task. Because an agent
    decision is a one-step routing/scheduling choice, the transition is
    terminal. Linking it to the next unrelated task would create an invalid
    Bellman target and potentially leak future observations.
    """
    by_task = {task.task_id: task for task in tasks}
    by_result = {result.task_id: result for result in results}
    if len(by_task) != len(tasks) or len(by_result) != len(results):
        raise ValueError("Duplicate task/result IDs within the episode")
    decisions = [task_id for task_id, _, _ in log.entries]
    if len(set(decisions)) != len(decisions):
        raise ValueError("Duplicate decision for a task within an agent tier")
    if any(task_id not in by_task or task_id not in by_result for task_id in decisions):
        raise ValueError("Missing task/result for at least one decision")

    rewards = []
    for task_id, state, action in log.entries:
        reward = team_reward(by_task[task_id], by_result[task_id])
        agent.observe(Transition(state.copy(), action, reward,
                                 np.zeros_like(state), True))
        rewards.append(reward)
    return float(np.mean(rewards)) if rewards else 0.0
