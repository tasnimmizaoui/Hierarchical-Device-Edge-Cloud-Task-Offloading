"""Small event-driven MEC backend used by Student 1.

The backend intentionally keeps the policy boundaries explicit:
the device chooses local versus edge, the edge chooses edge versus cloud,
and the cloud chooses the order in which queued cloud jobs execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import simpy

from mec_offloading.domain.actions import CloudSchedulingRule, DeviceAction, EdgeAction
from mec_offloading.domain.metrics import (
    cpu_energy_j,
    deadline_met,
    execution_time_s,
    radio_energy_j,
    transmission_time_s,
)
from mec_offloading.domain.models import (
    CloudState,
    ComputeNodeState,
    DeviceState,
    EdgeState,
    NetworkLinkState,
    QueueSummary,
    Task,
    TaskResult,
    Tier,
)
from mec_offloading.domain.ports import CloudSchedulingPolicy, DevicePolicy, EdgePolicy


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Resource capacities and link parameters for a reproducible scenario."""

    device_cpu_capacity_mips: float = 1_000.0
    edge_cpu_capacity_mips: float = 5_000.0
    cloud_cpu_capacity_mips: float = 20_000.0
    device_edge_bandwidth_mbps: float = 100.0
    edge_cloud_bandwidth_mbps: float = 1_000.0
    device_edge_propagation_s: float = 0.010
    edge_cloud_propagation_s: float = 0.040
    device_tx_power_w: float = 1.5
    device_rx_power_w: float = 0.5
    device_cpu_energy_per_mi_j: float = 0.001


@dataclass(slots=True)
class _CloudJob:
    task: Task
    queued_at_s: float
    finished: simpy.Event


class SimPyBackend:
    """A one-device, one-edge, one-cloud-node discrete-event backend."""

    def __init__(
        self,
        *,
        device: DeviceState | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.device = device or DeviceState(
            "device-1",
            self.config.device_cpu_capacity_mips,
            100.0,
            100.0,
            self.config.device_tx_power_w,
            self.config.device_rx_power_w,
            self.config.device_cpu_energy_per_mi_j,
        )

    def run_episode(
        self,
        tasks: list[Task] | tuple[Task, ...],
        device_policy: DevicePolicy,
        edge_policy: EdgePolicy,
        cloud_policy: CloudSchedulingPolicy,
        *,
        seed: int,
    ) -> list[TaskResult]:
        """Run tasks with deterministic ordering for a given input and seed."""
        del seed  # SimPy is deterministic here; stochastic policies own their seed.
        env = simpy.Environment()
        local = simpy.Resource(env, capacity=1)
        edge = simpy.Resource(env, capacity=1)
        link_up = simpy.Resource(env, capacity=1)
        link_down = simpy.Resource(env, capacity=1)
        edge_cloud_up = simpy.Resource(env, capacity=1)
        edge_cloud_down = simpy.Resource(env, capacity=1)
        cloud_queue: list[_CloudJob] = []
        cloud_wakeup = env.event()
        results: dict[str, TaskResult] = {}
        battery_j = self.device.battery_j

        def link_state(
            resource: simpy.Resource, bandwidth: float, propagation: float
        ) -> NetworkLinkState:
            queued = len(resource.queue)
            return NetworkLinkState(
                bandwidth,
                propagation,
                queue_wait_s=queued * propagation,
                utilization=float(resource.count > 0),
            )

        def edge_state() -> EdgeState:
            compute = ComputeNodeState(
                "edge-1",
                Tier.EDGE,
                self.config.edge_cpu_capacity_mips,
                cpu_utilization=float(edge.count > 0),
                queue_wait_s=len(edge.queue) / self.config.edge_cpu_capacity_mips,
                queue_length=len(edge.queue),
            )
            return EdgeState(
                compute,
                link_state(
                    link_up,
                    self.config.device_edge_bandwidth_mbps,
                    self.config.device_edge_propagation_s,
                ),
                link_state(
                    edge_cloud_up,
                    self.config.edge_cloud_bandwidth_mbps,
                    self.config.edge_cloud_propagation_s,
                ),
            )

        def cloud_state() -> CloudState:
            return CloudState(
                ComputeNodeState(
                    "cloud-1",
                    Tier.CLOUD,
                    self.config.cloud_cpu_capacity_mips,
                    cpu_utilization=0.0,
                    queue_wait_s=len(cloud_queue) / self.config.cloud_cpu_capacity_mips,
                    queue_length=len(cloud_queue),
                )
            )

        def transfer(
            resource: simpy.Resource,
            size_mb: float,
            bandwidth_mbps: float,
            propagation_s: float,
        ) -> Any:
            requested_at = env.now
            with resource.request() as request:
                yield request
                queue_wait = env.now - requested_at
                duration = transmission_time_s(
                    size_mb,
                    bandwidth_mbps,
                    propagation_s,
                    queue_wait,
                )
                yield env.timeout(duration - queue_wait)
                return duration, queue_wait

        def execute(resource: simpy.Resource, task: Task, capacity_mips: float) -> Any:
            requested_at = env.now
            with resource.request() as request:
                yield request
                queue_wait = env.now - requested_at
                duration = execution_time_s(task.required_mi, capacity_mips, queue_wait)
                yield env.timeout(duration - queue_wait)
                return duration, queue_wait

        def cloud_worker() -> Any:
            nonlocal cloud_wakeup
            while True:
                if not cloud_queue:
                    yield cloud_wakeup
                    cloud_wakeup = env.event()
                    continue
                summary = QueueSummary(
                    len(cloud_queue),
                    min(
                        job.task.absolute_deadline_s - env.now
                        for job in cloud_queue
                    ),
                    sum(job.task.absolute_deadline_s - env.now for job in cloud_queue)
                    / len(cloud_queue),
                    sum(job.task.required_mi for job in cloud_queue)
                    / self.config.cloud_cpu_capacity_mips
                    / len(cloud_queue),
                    max(env.now - job.task.arrival_time_s for job in cloud_queue),
                )
                rule = cloud_policy.choose_rule(summary, cloud_state())
                if rule == CloudSchedulingRule.EDF:
                    def key(job: _CloudJob) -> tuple[float, float]:
                        return (job.task.absolute_deadline_s, job.queued_at_s)
                elif rule == CloudSchedulingRule.SHORTEST_PROCESSING_TIME:
                    def key(job: _CloudJob) -> tuple[float, float]:
                        return (job.task.required_mi, job.queued_at_s)
                else:
                    def key(job: _CloudJob) -> tuple[float]:
                        return (job.queued_at_s,)
                job = min(cloud_queue, key=key)
                cloud_queue.remove(job)
                start = env.now
                queue_wait = start - job.queued_at_s
                yield env.timeout(job.task.required_mi / self.config.cloud_cpu_capacity_mips)
                job.finished.succeed((start, env.now, queue_wait))

        env.process(cloud_worker())

        def run_task(task: Task) -> Any:
            nonlocal battery_j
            yield env.timeout(max(0.0, task.arrival_time_s - env.now))
            current_device = DeviceState(
                self.device.device_id,
                self.device.cpu_capacity_mips,
                battery_j,
                self.device.battery_capacity_j,
                self.device.tx_power_w,
                self.device.rx_power_w,
                self.device.cpu_energy_per_mi_j,
            )
            selected_device_action = device_policy.choose(
                task, current_device, edge_state()
            )
            communication_delay = 0.0
            communication_bytes = 0
            energy = 0.0
            device_receive_duration = 0.0

            if selected_device_action == DeviceAction.EXECUTE_LOCAL:
                _, _ = yield from execute(local, task, self.device.cpu_capacity_mips)
                energy += cpu_energy_j(self.device.cpu_energy_per_mi_j, task.required_mi)
                route = (Tier.DEVICE,)
            else:
                upload, _ = yield from transfer(
                    link_up,
                    task.input_size_mb,
                    self.config.device_edge_bandwidth_mbps,
                    self.config.device_edge_propagation_s,
                )
                communication_delay += upload
                communication_bytes += int(round(task.input_size_mb * 1_000_000))
                energy += radio_energy_j(self.device.tx_power_w, upload)
                selected_edge_action = edge_policy.choose(task, edge_state(), cloud_state())
                if selected_edge_action == EdgeAction.EXECUTE_AT_EDGE:
                    yield from execute(edge, task, self.config.edge_cpu_capacity_mips)
                    route = (Tier.DEVICE, Tier.EDGE)
                else:
                    cloud_finished = env.event()
                    edge_upload, _ = yield from transfer(
                        edge_cloud_up,
                        task.input_size_mb,
                        self.config.edge_cloud_bandwidth_mbps,
                        self.config.edge_cloud_propagation_s,
                    )
                    communication_delay += edge_upload
                    communication_bytes += int(round(task.input_size_mb * 1_000_000))
                    cloud_queue.append(_CloudJob(task, env.now, cloud_finished))
                    if not cloud_wakeup.triggered:
                        cloud_wakeup.succeed()
                    _, _, _ = yield cloud_finished
                    route = (Tier.DEVICE, Tier.EDGE, Tier.CLOUD)
                download, _ = yield from transfer(
                    (
                        edge_cloud_down
                        if selected_edge_action == EdgeAction.FORWARD_TO_CLOUD
                        else link_down
                    ),
                    task.result_size_mb,
                    self.config.edge_cloud_bandwidth_mbps
                    if selected_edge_action == EdgeAction.FORWARD_TO_CLOUD
                    else self.config.device_edge_bandwidth_mbps,
                    self.config.edge_cloud_propagation_s
                    if selected_edge_action == EdgeAction.FORWARD_TO_CLOUD
                    else self.config.device_edge_propagation_s,
                )
                if selected_edge_action == EdgeAction.FORWARD_TO_CLOUD:
                    communication_delay += download
                    return_download, _ = yield from transfer(
                        link_down,
                        task.result_size_mb,
                        self.config.device_edge_bandwidth_mbps,
                        self.config.device_edge_propagation_s,
                    )
                    communication_delay += return_download
                    device_receive_duration = return_download
                else:
                    communication_delay += download
                    device_receive_duration = download
                communication_bytes += int(round(task.result_size_mb * 1_000_000))
                energy += radio_energy_j(self.device.rx_power_w, device_receive_duration)

            completion = env.now
            results[task.task_id] = TaskResult(
                task.task_id,
                route,
                task.arrival_time_s,
                completion,
                completion - task.arrival_time_s,
                deadline_met(completion, task.arrival_time_s, task.deadline_s),
                energy,
                communication_delay,
                communication_bytes,
            )

        for task in sorted(tasks, key=lambda item: (item.arrival_time_s, item.task_id)):
            env.process(run_task(task))
        env.run()
        return [results[task.task_id] for task in tasks]
