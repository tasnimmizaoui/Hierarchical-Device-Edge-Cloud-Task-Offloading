"""Baseline placement and scheduling policies."""

from mec_offloading.baselines.policies import (
    AlwaysEdgePolicy,
    DelayedGreedyComputeEdgePolicy,
    DelayedGreedyNearestDevicePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    FIFOCloudPolicy,
    GreedyNearestDevicePolicy,
    GreedyComputeEdgePolicy,
    LocalOnlyPolicy,
    RandomCloudPolicy,
    RandomDevicePolicy,
    RandomEdgePolicy,
)

__all__ = [
    "AlwaysEdgePolicy",
    "DelayedGreedyComputeEdgePolicy",
    "DelayedGreedyNearestDevicePolicy",
    "EDFCloudPolicy",
    "EdgeExecutePolicy",
    "EdgeForwardPolicy",
    "FIFOCloudPolicy",
    "GreedyNearestDevicePolicy",
    "GreedyComputeEdgePolicy",
    "LocalOnlyPolicy",
    "RandomCloudPolicy",
    "RandomDevicePolicy",
    "RandomEdgePolicy",
]
