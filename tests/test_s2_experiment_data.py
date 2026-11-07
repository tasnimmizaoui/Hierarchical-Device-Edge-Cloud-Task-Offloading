from __future__ import annotations

from mec_offloading.domain.models import Task
from mec_offloading.rl.experiment_data import (
    compress_arrivals,
    load_trace_experiment,
    offered_load,
    sample_tasks,
    topology_metadata,
)


def test_trace_splits_use_configured_multi_device_topology(tmp_path):
    task_path = tmp_path / "batch_task.csv"
    instance_path = tmp_path / "batch_instance.csv"
    topology_path = tmp_path / "topology.yaml"
    task_path.write_text(
        "".join(
            f"{i},{i + 1},job-{i},task-{i},1,Terminated,1,1\n"
            for i in range(12)
        ),
        encoding="utf-8",
    )
    instance_path.write_text(
        "".join(
            f"{i + 1},{i + 2},job-{i},task-{i},machine,Terminated,1,1,1,1,1,1\n"
            for i in range(12)
        ),
        encoding="utf-8",
    )
    topology_path.write_text(
        """topology:
  devices:
    - {id: device-1, cpu_mips: 1000, battery_j: 10, battery_capacity_j: 10,
       tx_power_w: 1, rx_power_w: 1, cpu_energy_per_mi_j: 0.001}
    - {id: device-2, cpu_mips: 1000, battery_j: 10, battery_capacity_j: 10,
       tx_power_w: 1, rx_power_w: 1, cpu_energy_per_mi_j: 0.001}
  edges:
    - {id: edge-1, cpu_mips: 2000, device_edge_bandwidth_mbps: 100,
       edge_cloud_bandwidth_mbps: 1000, device_edge_propagation_s: 0.01,
       edge_cloud_propagation_s: 0.04}
  device_to_edge: {device-1: edge-1, device-2: edge-1}
resources:
  cloud_cpu_mips: 5000
""",
        encoding="utf-8",
    )

    data = load_trace_experiment(
        task_path, instance_path, topology_path, max_tasks=12
    )
    relative_data = load_trace_experiment(
        task_path,
        instance_path,
        topology_path,
        max_tasks=12,
        deadline_policy="local-relative-2x",
    )
    best_tier_data = load_trace_experiment(
        task_path,
        instance_path,
        topology_path,
        max_tasks=12,
        deadline_policy="best-tier-calibrated",
    )

    assert [len(data.train), len(data.validation), len(data.test)] == [7, 2, 3]
    assert all(task.deadline_s == 5.0 for task in data.train)
    assert data.metadata["task_assumptions"]["deadline_s"] == 5.0
    assert data.metadata["task_assumptions"]["relative_deadline_multiplier"] is None
    capacities = {"device-1": 1000.0, "device-2": 1000.0}
    for task in relative_data.train:
        assert task.deadline_s == 2.0 * task.required_mi / capacities[task.device_id]
    assert (
        relative_data.metadata["task_assumptions"]["deadline_policy"]
        == "local-relative-2x"
    )
    assert relative_data.metadata["task_assumptions"]["deadline_s"] is None
    assert (
        relative_data.metadata["task_assumptions"]["relative_deadline_multiplier"]
        == 2.0
    )
    assert all(task.deadline_s == 1.0 for task in best_tier_data.train)
    assert (
        best_tier_data.metadata["task_assumptions"]["best_tier_deadline_multiplier"]
        == 0.9
    )
    assert (
        best_tier_data.metadata["task_assumptions"]["best_tier_deadline_slack_s"]
        == 0.5
    )
    assert best_tier_data.metadata["trace_adapter_version"] == (
        "alibaba2017-mean-completed-instance-lognormal-payload-v2"
    )
    all_ids = [{task.task_id for task in split} for split in
               (data.train, data.validation, data.test)]
    assert all_ids[0].isdisjoint(all_ids[1])
    assert all_ids[0].isdisjoint(all_ids[2])
    assert all_ids[1].isdisjoint(all_ids[2])
    assert {task.device_id for task in data.train} == {"device-1", "device-2"}
    assert data.metadata["edges"] == ["edge-1"]
    assert topology_metadata(data.simulation_config)["cloud_cpu_capacity_mips"] == 5000


def test_trace_task_sampling_is_seeded_and_without_replacement():
    tasks = tuple(
        Task(str(index), "device-1", float(index), 1, 100, .1, 2)
        for index in range(10)
    )

    first = sample_tasks(tasks, 4, 17)
    assert first == sample_tasks(tasks, 4, 17)
    assert len({task.task_id for task in first}) == 4
    assert sample_tasks(tasks, 20, 17) == list(tasks)
    assert sample_tasks(tasks, 4, 17) != sample_tasks(tasks, 4, 18)


def test_arrival_compression_scales_span_and_offered_load():
    tasks = (
        Task("a", "device-1", 10.0, 1, 100, .1, 2),
        Task("b", "device-1", 20.0, 1, 200, .1, 2),
    )
    compressed = compress_arrivals(tasks, .2)
    assert [task.arrival_time_s for task in compressed] == [10.0, 12.0]
    assert [task.arrival_time_s for task in tasks] == [10.0, 20.0]
    assert offered_load(compressed, 100.0) == 1.5
