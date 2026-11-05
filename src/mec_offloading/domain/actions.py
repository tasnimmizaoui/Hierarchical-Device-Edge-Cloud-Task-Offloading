"""Discrete action spaces shared by the simulator and RL adapters."""

from enum import IntEnum


class DeviceAction(IntEnum):
    EXECUTE_LOCAL = 0
    OFFLOAD_TO_EDGE = 1


class EdgeAction(IntEnum):
    EXECUTE_AT_EDGE = 0
    FORWARD_TO_CLOUD = 1


class CloudSchedulingRule(IntEnum):
    EDF = 0
    SHORTEST_PROCESSING_TIME = 1
    FIFO = 2
