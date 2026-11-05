"""Simple non-RL policies used for smoke tests and baseline experiments."""

from __future__ import annotations

import random

from mec_offloading.domain.actions import CloudSchedulingRule, DeviceAction, EdgeAction
from mec_offloading.domain.models import CloudState, DeviceState, EdgeState, QueueSummary, Task


class RandomDevicePolicy:
    def __init__(self, seed: int = 0) -> None:
        self._random = random.Random(seed)

    def choose(self, task: Task, device: DeviceState, edge: EdgeState) -> DeviceAction:
        del task, device, edge
        return DeviceAction(self._random.randrange(2))

class RandomEdgePolicy:
    def __init__(self, seed: int = 0) -> None:
        self._random = random.Random(seed)

    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        del task, edge, cloud
        return EdgeAction(self._random.randrange(2))


class RandomCloudPolicy:
    def __init__(self, seed: int = 0) -> None:
        self._random = random.Random(seed)

    def choose_rule(self, queue: QueueSummary, cloud: CloudState) -> CloudSchedulingRule:
        del queue, cloud
        return CloudSchedulingRule(self._random.randrange(3))


class GreedyNearestDevicePolicy:
    def choose(self, task: Task, device: DeviceState, edge: EdgeState) -> DeviceAction:
        local = task.required_mi / device.cpu_capacity_mips
        edge_estimate = (
            task.input_size_mb * 8 / edge.device_edge_link.bandwidth_mbps
            + edge.device_edge_link.propagation_delay_s
            + task.required_mi / edge.compute.cpu_capacity_mips
        )
        return DeviceAction.OFFLOAD_TO_EDGE if edge_estimate < local else DeviceAction.EXECUTE_LOCAL


class AlwaysEdgePolicy:
    def choose(self, task: Task, device: DeviceState, edge: EdgeState) -> DeviceAction:
        del task, device, edge
        return DeviceAction.OFFLOAD_TO_EDGE


class LocalOnlyPolicy:
    def choose(self, task: Task, device: DeviceState, edge: EdgeState) -> DeviceAction:
        del task, device, edge
        return DeviceAction.EXECUTE_LOCAL


class EdgeExecutePolicy:
    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        del task, edge, cloud
        return EdgeAction.EXECUTE_AT_EDGE


class EdgeForwardPolicy:
    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        del task, edge, cloud
        return EdgeAction.FORWARD_TO_CLOUD


class EDFCloudPolicy:
    def choose_rule(self, queue: QueueSummary, cloud: CloudState) -> CloudSchedulingRule:
        del queue, cloud
        return CloudSchedulingRule.EDF


class FIFOCloudPolicy:
    def choose_rule(self, queue: QueueSummary, cloud: CloudState) -> CloudSchedulingRule:
        del queue, cloud
        return CloudSchedulingRule.FIFO


class GreedyComputeEdgePolicy:
    """Route to cloud only if its estimated total completion delay is lower."""
    def choose(self, task: Task, edge: EdgeState, cloud: CloudState) -> EdgeAction:
        edge_delay = (edge.compute.queue_wait_s
                      + task.required_mi / edge.compute.cpu_capacity_mips)
        cloud_delay = (cloud.compute.queue_wait_s
                       + task.required_mi / cloud.compute.cpu_capacity_mips
                       + (task.input_size_mb + task.result_size_mb) * 8
                         / edge.edge_cloud_link.bandwidth_mbps
                       + 2 * edge.edge_cloud_link.propagation_delay_s)
        return (EdgeAction.FORWARD_TO_CLOUD if cloud_delay < edge_delay
                else EdgeAction.EXECUTE_AT_EDGE)
