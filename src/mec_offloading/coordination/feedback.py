"""Causal, adjacent-tier feedback snapshots with measurable control-plane cost.

A message sampled at t is *not observable* until t + propagation + serialization.
Delivery is lazy: pending messages are drained at the next agent decision time.
This gives correct decision-time causality without changing Student 1's SimPy
backend; feedback does NOT contend with data-plane links or block task execution.
"""
from __future__ import annotations

from dataclasses import dataclass

from mec_offloading.domain.models import NetworkLinkState


@dataclass(frozen=True)
class FeedbackCost:
    messages: int = 0
    bytes_sent: int = 0
    modeled_delay_s: float = 0.0


@dataclass(frozen=True)
class PendingFeedback:
    arrival_s: float
    value: float


class FeedbackCoordinator:
    DIRECTIONS = ("edge_to_device", "cloud_to_edge")

    def __init__(self, mode: str = "periodic", frequency: int = 5,
                 payload_bytes: int = 16, header_bytes: int = 16) -> None:
        if mode not in {"none", "periodic", "frequent"}:
            raise ValueError("mode must be none/periodic/frequent")
        if frequency < 1 or payload_bytes < 0 or header_bytes < 0:
            raise ValueError("invalid feedback configuration")
        self.mode, self.frequency = mode, frequency
        self.message_bytes = payload_bytes + header_bytes
        self._calls = dict.fromkeys(self.DIRECTIONS, 0)
        self._latest: dict[str, float] = {}
        self._pending: dict[str, list[PendingFeedback]] = {d: [] for d in self.DIRECTIONS}
        self._task_costs: dict[str, FeedbackCost] = {}
        self._last_time: dict[str, float] = {d: float("-inf") for d in self.DIRECTIONS}

    def observe(self, task_id: str, *, direction: str, value: float,
                link: NetworkLinkState, now_s: float = 0.0) -> float | None:
        if direction not in self._calls:
            raise ValueError(f"Unknown feedback direction: {direction}")
        if now_s < 0 or not (float("-inf") < now_s < float("inf")):
            raise ValueError("now_s must be finite and >= 0")
        if now_s < self._last_time[direction]:
            raise ValueError("Feedback calls must be chronological per direction")
        self._last_time[direction] = now_s

        # Deliver only messages whose simulated one-way delay has elapsed.
        arrived = [m for m in self._pending[direction] if m.arrival_s <= now_s]
        if arrived:
            self._latest[direction] = arrived[-1].value
        self._pending[direction] = [m for m in self._pending[direction]
                                    if m.arrival_s > now_s]
        self._calls[direction] += 1
        number = self._calls[direction]
        send = self.mode == "frequent" or (
            self.mode == "periodic" and (number - 1) % self.frequency == 0
        )
        if send:
            delay = link.propagation_delay_s + self.message_bytes * 8 / (
                link.bandwidth_mbps * 1e6
            )
            self._pending[direction].append(PendingFeedback(now_s + delay, float(value)))
            previous = self._task_costs.get(task_id, FeedbackCost())
            self._task_costs[task_id] = FeedbackCost(
                previous.messages + 1,
                previous.bytes_sent + self.message_bytes,
                previous.modeled_delay_s + delay,
            )
        return self._latest.get(direction)

    def cost(self, task_id: str) -> FeedbackCost:
        return self._task_costs.get(task_id, FeedbackCost())
