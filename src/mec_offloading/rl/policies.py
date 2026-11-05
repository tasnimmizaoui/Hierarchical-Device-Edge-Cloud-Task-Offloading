"""Inference/training adapters between Double-DQN and the frozen policy ports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.actions import CloudSchedulingRule, DeviceAction, EdgeAction
from mec_offloading.domain.models import (
    CloudState, DeviceState, EdgeState, QueueSummary, Task,
)
from mec_offloading.rl.observations import (
    centralized_observation, cloud_observation, device_observation, edge_observation,
)

if TYPE_CHECKING:
    from mec_offloading.rl.dqn import DQNAgent


@dataclass
class DecisionLog:
    entries: list[tuple[str, np.ndarray, int]] = field(default_factory=list)

    def add(self, task_id: str, observation: np.ndarray, action: int) -> None:
        self.entries.append((task_id, observation.copy(), int(action)))


class DeviceDQNPolicy:
    def __init__(self, agent: DQNAgent, feedback: FeedbackCoordinator,
                 *, training: bool = False) -> None:
        self.agent, self.feedback, self.training = agent, feedback, training
        self.clock_s = 0.0
        self.log = DecisionLog()

    def set_time(self, now_s: float) -> None:
        self.clock_s = now_s

    def choose(self, task: Task, device: DeviceState, edge: EdgeState) -> DeviceAction:
        congestion = self.feedback.observe(
            task.task_id, direction="edge_to_device",
            value=edge.compute.cpu_utilization, link=edge.device_edge_link,
            now_s=self.clock_s,
        )
        obs = device_observation(task, device, edge, self.clock_s, congestion)
        action = self.agent.act(obs, training=self.training)
        self.log.add(task.task_id, obs, action)
        return DeviceAction(action)


class EdgeDQNPolicy:
    def __init__(self, agent: DQNAgent, feedback: FeedbackCoordinator,
                 *, training: bool = False) -> None:
        self.agent, self.feedback, self.training = agent, feedback, training
        self.clock_s = 0.0
        self.log = DecisionLog()

    def set_time(self, now_s: float) -> None:
        self.clock_s = now_s

    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        congestion = self.feedback.observe(
            task.task_id, direction="cloud_to_edge",
            value=cloud.compute.cpu_utilization, link=edge.edge_cloud_link,
            now_s=self.clock_s,
        )
        obs = edge_observation(task, edge, self.clock_s, congestion)
        action = self.agent.act(obs, training=self.training)
        self.log.add(task.task_id, obs, action)
        return EdgeAction(action)


class CloudDQNPolicy:
    """Three actions: EDF=0, SPT=1, FIFO=2 (the base repo contract)."""
    def __init__(self, agent: DQNAgent, *, training: bool = False) -> None:
        self.agent, self.training = agent, training
        self.log = DecisionLog()
        self._pending: tuple[np.ndarray, int] | None = None

    def choose_rule(self, queue: QueueSummary, cloud: CloudState) -> CloudSchedulingRule:
        obs = cloud_observation(queue, cloud)
        action = self.agent.act(obs, training=self.training)
        self._pending = (obs, action)
        return CloudSchedulingRule(action)

    def record_dispatch(self, task_id: str) -> None:
        if self._pending is None:
            raise RuntimeError("No cloud scheduler decision to record")
        obs, action = self._pending
        self.log.add(task_id, obs, action)
        self._pending = None


class CentralizedDQNPolicy:
    """Centralized comparator: 0 device, 1 edge, 2 cloud with EDF.

    A single controller implements all three ports. Only its device decision
    accesses global state, via an optional backward-compatible backend hook.
    """
    def __init__(self, agent: DQNAgent, *, training: bool = False) -> None:
        self.agent, self.training = agent, training
        self.clock_s = 0.0
        self.log = DecisionLog()
        self._destinations: dict[str, int] = {}

    def set_time(self, now_s: float) -> None:
        self.clock_s = now_s

    def choose_global(self, task: Task, device: DeviceState, edge: EdgeState,
                      cloud: CloudState) -> DeviceAction:
        obs = centralized_observation(task, device, edge, cloud, self.clock_s)
        action = self.agent.act(obs, training=self.training)
        if action not in (0, 1, 2):
            raise ValueError("Centralized action must be 0..2")
        self._destinations[task.task_id] = action
        self.log.add(task.task_id, obs, action)
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
