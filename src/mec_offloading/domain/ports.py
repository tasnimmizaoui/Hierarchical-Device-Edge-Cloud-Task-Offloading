"""Policy and simulator protocols. Keep framework-specific imports out of this module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from mec_offloading.domain.actions import CloudSchedulingRule, DeviceAction, EdgeAction
from mec_offloading.domain.models import (
    CloudState,
    DeviceState,
    EdgeState,
    QueueSummary,
    Task,
    TaskResult,
)


class DevicePolicy(Protocol):
    def choose(
        self, task: Task, device: DeviceState, edge: EdgeState
    ) -> DeviceAction: ...


class EdgePolicy(Protocol):
    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction: ...


class CloudSchedulingPolicy(Protocol):
    def choose_rule(
        self, queue: QueueSummary, cloud: CloudState
    ) -> CloudSchedulingRule: ...


class SimulationBackend(Protocol):
    def run_episode(
        self,
        tasks: Sequence[Task],
        device_policy: DevicePolicy,
        edge_policy: EdgePolicy,
        cloud_policy: CloudSchedulingPolicy,
        *,
        seed: int,
    ) -> Sequence[TaskResult]: ...
