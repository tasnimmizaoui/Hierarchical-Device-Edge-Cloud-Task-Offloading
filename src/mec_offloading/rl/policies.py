"""Inference/training adapters between DQN agents and the frozen policy ports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.actions import CloudSchedulingRule, DeviceAction, EdgeAction
from mec_offloading.domain.models import (
    CloudState,
    DeviceState,
    EdgeState,
    QueueSummary,
    Task,
)
from mec_offloading.rl.observations import (
    centralized_observation,
    cloud_observation,
    device_observation,
    edge_observation,
)


class DecisionAgent(Protocol):
    def act(self, observation: np.ndarray, training: bool = False) -> int: ...


def _act_with_diagnostics(
    agent: DecisionAgent,
    observation: np.ndarray,
    training: bool,
) -> tuple[int, np.ndarray | None, float]:
    diagnose = getattr(agent, "act_with_diagnostics", None)
    if callable(diagnose):
        return diagnose(observation, training=training)
    action = agent.act(observation, training=training)
    epsilon = agent.epsilon() if callable(getattr(agent, "epsilon", None)) else 0.0
    return action, None, epsilon


@dataclass
class DecisionLog:
    entries: list[tuple[str, np.ndarray, int]] = field(default_factory=list)
    contested_actions: dict[CloudSchedulingRule, int] = field(default_factory=dict)
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    dispatch_rewards: dict[str, float] = field(default_factory=dict)

    def add(self, task_id: str, observation: np.ndarray, action: int,
            q_values: np.ndarray | None = None, epsilon: float = 0.0) -> None:
        self.entries.append((task_id, observation.copy(), int(action)))
        self.diagnostics.append({
            "task_id": task_id,
            "observation": observation.copy(),
            "action": int(action),
            "q_values": None if q_values is None else q_values.copy(),
            "epsilon": float(epsilon),
        })


class DeviceDQNPolicy:
    def __init__(self, agent: DecisionAgent, feedback: FeedbackCoordinator,
                 *, training: bool = False) -> None:
        self.agent, self.feedback, self.training = agent, feedback, training
        self.clock_s = 0.0
        self.log = DecisionLog()

    def set_time(self, now_s: float) -> None:
        self.clock_s = now_s

    def choose(self, task: Task, device: DeviceState, edge: EdgeState) -> DeviceAction:
        congestion = self.feedback.observe(
            task.task_id, direction="edge_to_device",
            value=edge.compute.queue_wait_s / (edge.compute.queue_wait_s + 1.0),
            link=edge.device_edge_link,
            now_s=self.clock_s,
            scope=f"{edge.compute.node_id}->{device.device_id}",
        )
        obs = device_observation(task, device, edge, self.clock_s, congestion)
        action, q_values, epsilon = _act_with_diagnostics(self.agent, obs, self.training)
        self.log.add(task.task_id, obs, action, q_values, epsilon)
        return DeviceAction(action)


class EdgeDQNPolicy:
    def __init__(self, agent: DecisionAgent, feedback: FeedbackCoordinator,
                 *, training: bool = False) -> None:
        self.agent, self.feedback, self.training = agent, feedback, training
        self.clock_s = 0.0
        self.log = DecisionLog()

    def set_time(self, now_s: float) -> None:
        self.clock_s = now_s

    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        congestion = self.feedback.observe(
            task.task_id, direction="cloud_to_edge",
            value=cloud.compute.queue_wait_s / (cloud.compute.queue_wait_s + 1.0),
            link=edge.edge_cloud_link,
            now_s=self.clock_s,
            scope=f"{cloud.compute.node_id}->{edge.compute.node_id}",
        )
        obs = edge_observation(task, edge, self.clock_s, congestion)
        action, q_values, epsilon = _act_with_diagnostics(self.agent, obs, self.training)
        self.log.add(task.task_id, obs, action, q_values, epsilon)
        return EdgeAction(action)


class CloudDQNPolicy:
    """Three actions: EDF=0, SPT=1, FIFO=2 (the base repo contract)."""
    def __init__(self, agent: DecisionAgent, *, training: bool = False) -> None:
        self.agent, self.training = agent, training
        self.log = DecisionLog()
        self._pending: tuple[np.ndarray, int] | None = None
        self._pending_info: tuple[np.ndarray | None, float, int] | None = None

    def choose_rule(self, queue: QueueSummary, cloud: CloudState) -> CloudSchedulingRule:
        if queue.queue_length <= 1:
            self._pending = None
            return CloudSchedulingRule.EDF
        obs = cloud_observation(queue, cloud)
        action, q_values, epsilon = _act_with_diagnostics(self.agent, obs, self.training)
        self._pending = (obs, action)
        rule = CloudSchedulingRule(action)
        counts = self.log.contested_actions
        counts[rule] = counts.get(rule, 0) + 1
        self._pending_info = (q_values, epsilon, queue.queue_length)
        return rule

    def record_dispatch(self, task_id: str, estimated_service_s: float = 0.0) -> None:
        if self._pending is None:
            return
        obs, action = self._pending
        if self._pending_info is None:
            raise RuntimeError("Cloud decision diagnostics are missing")
        q_values, epsilon, queue_length = self._pending_info
        self.log.add(task_id, obs, action, q_values, epsilon)
        queue_delay_cost = queue_length * max(0.0, estimated_service_s)
        self.log.dispatch_rewards[task_id] = -float(
            min(1.0, np.log1p(queue_delay_cost) / np.log1p(10_000.0))
        )
        self._pending = None
        self._pending_info = None


class CentralizedDQNPolicy:
    """Centralized comparator: 0 device, 1 edge, 2 cloud with EDF.

    A single controller implements all three ports. Only its device decision
    accesses global state, via an optional backward-compatible backend hook.
    """
    def __init__(self, agent: DecisionAgent, *, training: bool = False) -> None:
        self.agent, self.training = agent, training
        self.clock_s = 0.0
        self.log = DecisionLog()
        self._destinations: dict[str, int] = {}

    def set_time(self, now_s: float) -> None:
        self.clock_s = now_s

    def choose_global(self, task: Task, device: DeviceState, edge: EdgeState,
                      cloud: CloudState) -> DeviceAction:
        obs = centralized_observation(task, device, edge, cloud, self.clock_s)
        action, q_values, epsilon = _act_with_diagnostics(self.agent, obs, self.training)
        if action not in (0, 1, 2):
            raise ValueError("Centralized action must be 0..2")
        self._destinations[task.task_id] = action
        self.log.add(task.task_id, obs, action, q_values, epsilon)
        return DeviceAction.EXECUTE_LOCAL if action == 0 else DeviceAction.OFFLOAD_TO_EDGE

    def choose(self, task: Task, device: DeviceState, edge: EdgeState) -> DeviceAction:
        raise RuntimeError("Centralized comparator needs backend global-state hook")

    def choose_edge(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        del edge, cloud
        return (EdgeAction.FORWARD_TO_CLOUD if self._destinations[task.task_id] == 2
                else EdgeAction.EXECUTE_AT_EDGE)

    def choose_rule(self, queue: QueueSummary, cloud: CloudState) -> CloudSchedulingRule:
        del queue, cloud
        return CloudSchedulingRule.EDF


class CentralizedEdgeAdapter:
    """Adapts centralized controller's edge decision to EdgePolicy protocol."""
    def __init__(self, controller: CentralizedDQNPolicy) -> None:
        self.controller = controller

    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        return self.controller.choose_edge(task, edge, cloud)
