"""Per-task terminal credit and explicit normalized multi-objective reward.

This module deliberately has no SimPy dependency, so reward and credit
semantics can be unit-tested while the simulator integration is pending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mec_offloading.domain.metrics import execution_time_s, transmission_time_s
from mec_offloading.domain.models import Task, TaskResult
from mec_offloading.rl.dqn import Transition
from mec_offloading.rl.policies import DecisionLog
from mec_offloading.simulation.simpy_backend import SimPyBackend

REWARD_VERSION = "feasibility-aware-energy-cloud-queue-cost-v5"


class ReplayAgent(Protocol):
    def observe(self, transition: Transition) -> float | None: ...


@dataclass(frozen=True)
class RewardWeights:
    latency: float = 0.35
    device_energy: float = 0.25
    deadline_miss: float = 0.35
    coordination: float = 0.05
    energy_reference_j: float = 0.3
    communication_reference_bytes: float = 64.0


def minimum_possible_latency_s(task: Task, backend: SimPyBackend) -> float:
    """Best no-queue latency among local, edge and cloud routes."""
    devices = {device.device_id: device for device in backend.topology.devices}
    device = devices[task.device_id]
    edge = backend.topology.edge_for(task.device_id)

    local_s = execution_time_s(task.required_mi, device.cpu_capacity_mips)
    edge_s = (
        transmission_time_s(
            task.input_size_mb,
            edge.device_edge_bandwidth_mbps,
            edge.device_edge_propagation_s,
        )
        + execution_time_s(task.required_mi, edge.cpu_capacity_mips)
        + transmission_time_s(
            task.result_size_mb,
            edge.device_edge_bandwidth_mbps,
            edge.device_edge_propagation_s,
        )
    )
    cloud_s = (
        transmission_time_s(
            task.input_size_mb,
            edge.device_edge_bandwidth_mbps,
            edge.device_edge_propagation_s,
        )
        + transmission_time_s(
            task.input_size_mb,
            edge.edge_cloud_bandwidth_mbps,
            edge.edge_cloud_propagation_s,
        )
        + execution_time_s(task.required_mi, backend.config.cloud_cpu_capacity_mips)
        + transmission_time_s(
            task.result_size_mb,
            edge.edge_cloud_bandwidth_mbps,
            edge.edge_cloud_propagation_s,
        )
        + transmission_time_s(
            task.result_size_mb,
            edge.device_edge_bandwidth_mbps,
            edge.device_edge_propagation_s,
        )
    )
    return min(local_s, edge_s, cloud_s)


def deadline_feasibility(
    tasks: list[Task] | tuple[Task, ...],
    backend: SimPyBackend,
) -> dict[str, bool]:
    """Whether any route can meet each task deadline with empty queues."""
    return {
        task.task_id: minimum_possible_latency_s(task, backend) <= task.deadline_s
        for task in tasks
    }


def team_reward(
    task: Task,
    result: TaskResult,
    weights: RewardWeights | None = None,
    *,
    deadline_feasible: bool = True,
) -> float:
    """Shared per-task terminal reward; bounded dimensionless penalty."""
    weights = weights or RewardWeights()
    if weights.energy_reference_j <= 0 or weights.communication_reference_bytes <= 0:
        raise ValueError("Reward normalization references must be positive")
    if min(weights.latency, weights.device_energy, weights.deadline_miss,
           weights.coordination) < 0:
        raise ValueError("Reward weights must be nonnegative")
    latency_ratio = result.end_to_end_latency_s / task.deadline_s
    latency_fraction = min(np.log1p(latency_ratio), np.log1p(10_000.0)) / np.log1p(
        10_000.0
    )
    energy_fraction = min(3.0, result.device_energy_j / weights.energy_reference_j)
    communication_fraction = min(3.0, result.feedback_bytes /
                                 weights.communication_reference_bytes)
    return -float(weights.latency * latency_fraction
                  + weights.device_energy * energy_fraction
                  + weights.deadline_miss * (
                      deadline_feasible and not result.deadline_met
                  )
                  + weights.coordination * communication_fraction)


def update_dqn_from_episode(agent: ReplayAgent, log: DecisionLog,
                            tasks: list[Task], results: list[TaskResult],
                            feasibility: dict[str, bool] | None = None,
                            *, credit_mode: str = "terminal",
                            weights: RewardWeights | None = None) -> float:
    """Assign terminal credit to decisions only after the full episode ends.

    ``next-decision`` is an explicit ablation: transitions bootstrap from the
    same agent's next logged decision, with the last transition terminal.
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
    if not set(log.dispatch_rewards).issubset(decisions):
        raise ValueError("Dispatch reward exists without a corresponding decision")
    if credit_mode not in {"terminal", "next-decision"}:
        raise ValueError("credit_mode must be 'terminal' or 'next-decision'")

    rewards = []
    for index, (task_id, state, action) in enumerate(log.entries):
        reward = log.dispatch_rewards.get(task_id)
        if reward is None:
            reward = team_reward(
                by_task[task_id],
                by_result[task_id],
                weights,
                deadline_feasible=(
                    feasibility.get(task_id, True) if feasibility is not None else True
                ),
            )
        if credit_mode == "next-decision" and index + 1 < len(log.entries):
            next_state = log.entries[index + 1][1].copy()
            done = False
        else:
            next_state = np.zeros_like(state)
            done = True
        agent.observe(Transition(state.copy(), action, reward, next_state, done))
        rewards.append(reward)
    return float(np.mean(rewards)) if rewards else 0.0
