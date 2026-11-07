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
    information_age_s: float = 0.0
    information_age_samples: int = 0


@dataclass(frozen=True)
class PendingFeedback:
    arrival_s: float
    generated_s: float
    value: float


class FeedbackCoordinator:
    DIRECTIONS = ("edge_to_device", "cloud_to_edge")
    MODES = ("none", "periodic", "frequent", "event")
    EVENT_SIGNAL_CHANGE = 0.25

    def __init__(self, mode: str = "periodic", frequency: int = 5,
                 payload_bytes: int = 16, header_bytes: int = 16,
                 event_threshold: float = EVENT_SIGNAL_CHANGE) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if (
            frequency < 1 or payload_bytes < 0 or header_bytes < 0
            or not 0 <= event_threshold <= 1
        ):
            raise ValueError("invalid feedback configuration")
        self.mode, self.frequency = mode, frequency
        self.event_threshold = event_threshold
        self.message_bytes = payload_bytes + header_bytes
        self._calls: dict[tuple[str, str], int] = {}
        self._latest: dict[tuple[str, str], float] = {}
        self._latest_generated: dict[tuple[str, str], float] = {}
        self._pending: dict[tuple[str, str], list[PendingFeedback]] = {}
        self._task_costs: dict[str, FeedbackCost] = {}
        self._last_time: dict[tuple[str, str], float] = {}
        self._last_sent: dict[tuple[str, str], tuple[int, float]] = {}

    def observe(self, task_id: str, *, direction: str, value: float,
                link: NetworkLinkState, now_s: float = 0.0,
                scope: str = "default") -> float | None:
        if direction not in self.DIRECTIONS:
            raise ValueError(f"Unknown feedback direction: {direction}")
        if not scope:
            raise ValueError("scope must not be empty")
        if now_s < 0 or not (float("-inf") < now_s < float("inf")):
            raise ValueError("now_s must be finite and >= 0")
        channel = (direction, scope)
        if now_s < self._last_time.get(channel, float("-inf")):
            raise ValueError("Feedback calls must be chronological per channel")
        self._last_time[channel] = now_s

        # Deliver only messages whose simulated one-way delay has elapsed.
        pending = self._pending.setdefault(channel, [])
        arrived = [m for m in pending if m.arrival_s <= now_s]
        if arrived:
            newest = arrived[-1]
            self._latest[channel] = newest.value
            self._latest_generated[channel] = newest.generated_s
        self._pending[channel] = [m for m in pending if m.arrival_s > now_s]
        if channel in self._latest_generated:
            previous = self._task_costs.get(task_id, FeedbackCost())
            self._task_costs[task_id] = FeedbackCost(
                previous.messages,
                previous.bytes_sent,
                previous.modeled_delay_s,
                previous.information_age_s
                + max(0.0, now_s - self._latest_generated[channel]),
                previous.information_age_samples + 1,
            )
        number = self._calls.get(channel, 0) + 1
        self._calls[channel] = number
        send = self.mode == "frequent" or (
            self.mode == "periodic" and (number - 1) % self.frequency == 0
        )
        if self.mode == "event":
            last = self._last_sent.get(channel)
            send = (
                last is None
                or number - last[0] >= self.frequency
                or abs(value - last[1]) >= self.event_threshold
            )
        if send:
            if self.mode == "event":
                self._last_sent[channel] = (number, value)
            delay = link.propagation_delay_s + self.message_bytes * 8 / (
                link.bandwidth_mbps * 1e6
            )
            self._pending[channel].append(
                PendingFeedback(now_s + delay, now_s, float(value))
            )
            previous = self._task_costs.get(task_id, FeedbackCost())
            self._task_costs[task_id] = FeedbackCost(
                previous.messages + 1,
                previous.bytes_sent + self.message_bytes,
                previous.modeled_delay_s + delay,
                previous.information_age_s,
                previous.information_age_samples,
            )
        return self._latest.get(channel)

    def cost(self, task_id: str) -> FeedbackCost:
        return self._task_costs.get(task_id, FeedbackCost())
