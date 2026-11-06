"""Event-driven MEC backend with configurable devices, edges, and one cloud."""

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
class EdgeConfig:
    """One independently queued edge server and its network links."""

    edge_id: str
    cpu_capacity_mips: float
    device_edge_bandwidth_mbps: float = 100.0
    edge_cloud_bandwidth_mbps: float = 1_000.0
    device_edge_propagation_s: float = 0.010
    edge_cloud_propagation_s: float = 0.040


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    """Devices, edges, and fixed device-to-edge associations."""

    devices: tuple[DeviceState, ...]
    edges: tuple[EdgeConfig, ...]
    device_to_edge: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        device_ids = {device.device_id for device in self.devices}
        edge_ids = {edge.edge_id for edge in self.edges}
        associations = dict(self.device_to_edge)
        if not device_ids or len(device_ids) != len(self.devices):
            raise ValueError("topology devices must have unique IDs")
        if not edge_ids or len(edge_ids) != len(self.edges):
            raise ValueError("topology edges must have unique IDs")
        if set(associations) != device_ids or set(associations.values()) - edge_ids:
            raise ValueError("every device must have one valid edge association")

    def edge_for(self, device_id: str) -> EdgeConfig:
        edge_id = dict(self.device_to_edge).get(device_id)
        if edge_id is None:
            raise ValueError(f"no edge association for device {device_id}")
        return next(edge for edge in self.edges if edge.edge_id == edge_id)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Resource capacities and optional topology parameters."""

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
    topology: TopologyConfig | None = None


@dataclass(slots=True)
class _CloudJob:
    task: Task
    edge_id: str
    queued_at_s: float
    finished: simpy.Event


class SimPyBackend:
    """SimPy backend preserving independent device, edge, and cloud policies."""

    def __init__(
        self,
        *,
        device: DeviceState | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        default_device = device or DeviceState(
            "device-1",
            self.config.device_cpu_capacity_mips,
            100.0,
            100.0,
            self.config.device_tx_power_w,
            self.config.device_rx_power_w,
            self.config.device_cpu_energy_per_mi_j,
        )
        self.topology = self.config.topology or TopologyConfig(
            (default_device,),
            (
                EdgeConfig(
                    "edge-1",
                    self.config.edge_cpu_capacity_mips,
                    self.config.device_edge_bandwidth_mbps,
                    self.config.edge_cloud_bandwidth_mbps,
                    self.config.device_edge_propagation_s,
                    self.config.edge_cloud_propagation_s,
                ),
            ),
            ((default_device.device_id, "edge-1"),),
        )
        self.device = default_device

    def run_episode(
        self,
        tasks: list[Task] | tuple[Task, ...],
        device_policy: DevicePolicy,
        edge_policy: EdgePolicy,
        cloud_policy: CloudSchedulingPolicy,
        *,
        seed: int,
    ) -> list[TaskResult]:
        """Run tasks with fresh resources and deterministic event ordering."""
        del seed
        known_devices = {device.device_id: device for device in self.topology.devices}
        for task in tasks:
            if task.device_id not in known_devices:
                raise ValueError(f"task {task.task_id} references unknown device {task.device_id}")
        env = simpy.Environment()
        local = {device_id: simpy.Resource(env, capacity=1) for device_id in known_devices}
        edge_resources = {
            edge.edge_id: simpy.Resource(env, capacity=1) for edge in self.topology.edges
        }
        associations = {
            device_id: self.topology.edge_for(device_id) for device_id in known_devices
        }
        uplink = {
            (device_id, edge.edge_id): simpy.Resource(env, capacity=1)
            for device_id, edge in associations.items()
        }
        downlink = {
            (device_id, edge.edge_id): simpy.Resource(env, capacity=1)
            for device_id, edge in associations.items()
        }
        cloud_uplink = {
            edge.edge_id: simpy.Resource(env, capacity=1) for edge in self.topology.edges
        }
        cloud_downlink = {
            edge.edge_id: simpy.Resource(env, capacity=1) for edge in self.topology.edges
        }
        cloud_queue: list[_CloudJob] = []
        cloud_wakeup = env.event()
        cloud_busy = False
        results: dict[str, TaskResult] = {}
        battery_j = {device_id: device.battery_j for device_id, device in known_devices.items()}

        def link_state(
            resource: simpy.Resource, bandwidth: float, propagation: float
        ) -> NetworkLinkState:
            return NetworkLinkState(
                bandwidth,
                propagation,
                queue_wait_s=len(resource.queue) * propagation,
                utilization=float(resource.count > 0),
            )

        def edge_state(device_id: str) -> EdgeState:
            edge = associations[device_id]
            return EdgeState(
                ComputeNodeState(
                    edge.edge_id,
                    Tier.EDGE,
                    edge.cpu_capacity_mips,
                    cpu_utilization=float(edge_resources[edge.edge_id].count > 0),
                    queue_wait_s=len(edge_resources[edge.edge_id].queue)
                    / edge.cpu_capacity_mips,
                    queue_length=len(edge_resources[edge.edge_id].queue),
                ),
                link_state(
                    uplink[(device_id, edge.edge_id)],
                    edge.device_edge_bandwidth_mbps,
                    edge.device_edge_propagation_s,
                ),
                link_state(
                    cloud_uplink[edge.edge_id],
                    edge.edge_cloud_bandwidth_mbps,
                    edge.edge_cloud_propagation_s,
                ),
            )

        def cloud_state() -> CloudState:
            return CloudState(
                ComputeNodeState(
                    "cloud-1",
                    Tier.CLOUD,
                    self.config.cloud_cpu_capacity_mips,
                    cpu_utilization=float(cloud_busy),
                    queue_wait_s=len(cloud_queue) / self.config.cloud_cpu_capacity_mips,
                    queue_length=len(cloud_queue),
                )
            )

        def transfer(
            resource: simpy.Resource, size_mb: float, bandwidth: float, propagation: float
        ) -> Any:
            requested_at = env.now
            with resource.request() as request:
                yield request
                queue_wait = env.now - requested_at
                duration = transmission_time_s(size_mb, bandwidth, propagation, queue_wait)
                yield env.timeout(duration - queue_wait)
                return duration, queue_wait

        def execute(resource: simpy.Resource, task: Task, capacity: float) -> Any:
            requested_at = env.now
            with resource.request() as request:
                yield request
                started_at = env.now
                queue_wait = started_at - requested_at
                duration = execution_time_s(task.required_mi, capacity, queue_wait)
                yield env.timeout(duration - queue_wait)
                return started_at

        def cloud_worker() -> Any:
            nonlocal cloud_wakeup, cloud_busy
            while True:
                if not cloud_queue:
                    yield cloud_wakeup
                    cloud_wakeup = env.event()
                    continue
                summary = QueueSummary(
                    len(cloud_queue),
                    min(job.task.absolute_deadline_s - env.now for job in cloud_queue),
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
                dispatch_hook = getattr(cloud_policy, "record_dispatch", None)
                if callable(dispatch_hook):
                    dispatch_hook(job.task.task_id)
                start = env.now
                cloud_busy = True
                yield env.timeout(job.task.required_mi / self.config.cloud_cpu_capacity_mips)
                cloud_busy = False
                job.finished.succeed((start, env.now))

        env.process(cloud_worker())

        def run_task(task: Task) -> Any:
            device = known_devices[task.device_id]
            edge = associations[task.device_id]
            yield env.timeout(max(0.0, task.arrival_time_s - env.now))
            current_device = DeviceState(
                device.device_id,
                device.cpu_capacity_mips,
                battery_j[device.device_id],
                device.battery_capacity_j,
                device.tx_power_w,
                device.rx_power_w,
                device.cpu_energy_per_mi_j,
            )
            set_clock = getattr(device_policy, "set_time", None)
            if callable(set_clock):
                set_clock(env.now)
            current_edge = edge_state(device.device_id)
            global_choice = getattr(device_policy, "choose_global", None)
            if callable(global_choice):
                selected = global_choice(task, current_device, current_edge, cloud_state())
            else:
                selected = device_policy.choose(task, current_device, current_edge)
            communication_delay = 0.0
            communication_bytes = 0
            energy = 0.0
            execution_start = env.now
            if selected == DeviceAction.EXECUTE_LOCAL:
                execution_start = yield from execute(
                    local[device.device_id], task, device.cpu_capacity_mips
                )
                energy = cpu_energy_j(device.cpu_energy_per_mi_j, task.required_mi)
                route = (Tier.DEVICE,)
                source_edge_id = None
            else:
                upload, _ = yield from transfer(
                    uplink[(device.device_id, edge.edge_id)],
                    task.input_size_mb,
                    edge.device_edge_bandwidth_mbps,
                    edge.device_edge_propagation_s,
                )
                communication_delay += upload
                communication_bytes += int(round(task.input_size_mb * 1_000_000))
                energy += radio_energy_j(device.tx_power_w, upload)
                edge_policy_clock = getattr(edge_policy, "set_time", None)
                if callable(edge_policy_clock):
                    edge_policy_clock(env.now)
                current_edge = edge_state(device.device_id)
                edge_choice = getattr(edge_policy, "choose_edge", None)
                if callable(edge_choice):
                    edge_action = edge_choice(task, current_edge, cloud_state())
                else:
                    edge_action = edge_policy.choose(task, current_edge, cloud_state())
                source_edge_id = edge.edge_id
                if edge_action == EdgeAction.EXECUTE_AT_EDGE:
                    execution_start = yield from execute(
                        edge_resources[edge.edge_id], task, edge.cpu_capacity_mips
                    )
                    route = (Tier.DEVICE, Tier.EDGE)
                    return_resource = downlink[(device.device_id, edge.edge_id)]
                    return_bandwidth = edge.device_edge_bandwidth_mbps
                    return_propagation = edge.device_edge_propagation_s
                else:
                    finished = env.event()
                    edge_upload, _ = yield from transfer(
                        cloud_uplink[edge.edge_id],
                        task.input_size_mb,
                        edge.edge_cloud_bandwidth_mbps,
                        edge.edge_cloud_propagation_s,
                    )
                    communication_delay += edge_upload
                    communication_bytes += int(round(task.input_size_mb * 1_000_000))
                    cloud_queue.append(_CloudJob(task, edge.edge_id, env.now, finished))
                    if not cloud_wakeup.triggered:
                        cloud_wakeup.succeed()
                    execution_start, _ = yield finished
                    route = (Tier.DEVICE, Tier.EDGE, Tier.CLOUD)
                    cloud_download, _ = yield from transfer(
                        cloud_downlink[edge.edge_id],
                        task.result_size_mb,
                        edge.edge_cloud_bandwidth_mbps,
                        edge.edge_cloud_propagation_s,
                    )
                    communication_delay += cloud_download
                    communication_bytes += int(round(task.result_size_mb * 1_000_000))
                    return_resource = downlink[(device.device_id, edge.edge_id)]
                    return_bandwidth = edge.device_edge_bandwidth_mbps
                    return_propagation = edge.device_edge_propagation_s
                result_transfer, _ = yield from transfer(
                    return_resource,
                    task.result_size_mb,
                    return_bandwidth,
                    return_propagation,
                )
                communication_delay += result_transfer
                communication_bytes += int(round(task.result_size_mb * 1_000_000))
                energy += radio_energy_j(device.rx_power_w, result_transfer)
            completion = env.now
            battery_j[device.device_id] = max(0.0, battery_j[device.device_id] - energy)
            feedback = getattr(device_policy, "feedback", None)
            feedback_cost = (
                feedback.cost(task.task_id)
                if feedback is not None and callable(getattr(feedback, "cost", None))
                else None
            )
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
                execution_start_time_s=execution_start,
                feedback_messages=feedback_cost.messages if feedback_cost else 0,
                feedback_bytes=feedback_cost.bytes_sent if feedback_cost else 0,
                feedback_modeled_delay_s=(
                    feedback_cost.modeled_delay_s if feedback_cost else 0.0
                ),
                source_device_id=device.device_id,
                source_edge_id=source_edge_id,
            )

        for task in sorted(tasks, key=lambda item: (item.arrival_time_s, item.task_id)):
            env.process(run_task(task))
        env.run()
        return [results[task.task_id] for task in tasks]
