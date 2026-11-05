"""Baseline placement and scheduling policies."""

from mec_offloading.baselines.policies import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    FIFOCloudPolicy,
    GreedyNearestDevicePolicy,
    LocalOnlyPolicy,
    RandomCloudPolicy,
    RandomDevicePolicy,
    RandomEdgePolicy,
)

__all__ = [
    "AlwaysEdgePolicy",
    "EDFCloudPolicy",
    "EdgeExecutePolicy",
    "EdgeForwardPolicy",
    "FIFOCloudPolicy",
    "GreedyNearestDevicePolicy",
    "LocalOnlyPolicy",
    "RandomCloudPolicy",
    "RandomDevicePolicy",
    "RandomEdgePolicy",
]
