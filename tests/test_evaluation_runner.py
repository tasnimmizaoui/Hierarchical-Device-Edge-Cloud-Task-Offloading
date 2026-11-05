import csv

from mec_offloading.baselines import (
    AlwaysEdgePolicy,
    EDFCloudPolicy,
    EdgeExecutePolicy,
    EdgeForwardPolicy,
    FIFOCloudPolicy,
    LocalOnlyPolicy,
)
from mec_offloading.domain.models import Task
from mec_offloading.evaluation import (
    BaselinePolicy,
    aggregate_results,
    run_baselines,
)
from mec_offloading.simulation import SimPyBackend, SimulationConfig
from mec_offloading.workloads import generate_synthetic_tasks


def test_aggregate_results_calculates_required_metrics():
    tasks = generate_synthetic_tasks(count=2, seed=4)
    results = list(
        SimPyBackend().run_episode(
            tasks, LocalOnlyPolicy(), EdgeExecutePolicy(), EDFCloudPolicy(), seed=4
        )
    )

    aggregate = aggregate_results("local", results)

    assert aggregate["task_count"] == 2
    assert aggregate["mean_latency_s"] > 0
    assert aggregate["median_latency_s"] > 0
    assert aggregate["p95_latency_s"] >= aggregate["median_latency_s"]
    assert 0.0 <= aggregate["deadline_success_rate"] <= 1.0
    assert aggregate["deadline_success_rate"] + aggregate["deadline_miss_rate"] == 1.0


def test_run_baselines_writes_detailed_and_aggregate_csv(tmp_path):
    tasks = generate_synthetic_tasks(count=2, seed=5)
    detailed, aggregate, _ = run_baselines(
        {5: tasks},
        [BaselinePolicy("local", LocalOnlyPolicy(), EdgeExecutePolicy(), EDFCloudPolicy())],
        output_dir=tmp_path,
    )

    with detailed.open(newline="", encoding="utf-8") as file:
        detailed_rows = list(csv.DictReader(file))
    with aggregate.open(newline="", encoding="utf-8") as file:
        aggregate_rows = list(csv.DictReader(file))

    assert len(detailed_rows) == 2
    assert len(aggregate_rows) == 1
    assert aggregate_rows[0]["policy"] == "local"
    assert aggregate_rows[0]["seed"] == "5"


def test_same_seed_recreates_workload_and_different_seed_changes_it():
    first = generate_synthetic_tasks(count=4, seed=13)
    repeat = generate_synthetic_tasks(count=4, seed=13)
    different = generate_synthetic_tasks(count=4, seed=14)

    assert first == repeat
    assert first != different


def test_multi_seed_runner_records_seed_and_statistics(tmp_path):
    tasks_by_seed = {
        seed: generate_synthetic_tasks(count=2, seed=seed)
        for seed in (1, 2, 3)
    }

    detailed, aggregate, statistics = run_baselines(
        tasks_by_seed,
        lambda seed: [
            BaselinePolicy(
                "local",
                LocalOnlyPolicy(),
                EdgeExecutePolicy(),
                EDFCloudPolicy(),
            )
        ],
        output_dir=tmp_path,
    )

    with detailed.open(newline="", encoding="utf-8") as file:
        detailed_rows = list(csv.DictReader(file))
    with aggregate.open(newline="", encoding="utf-8") as file:
        aggregate_rows = list(csv.DictReader(file))
    with statistics.open(newline="", encoding="utf-8") as file:
        statistics_rows = list(csv.DictReader(file))

    assert {row["seed"] for row in detailed_rows} == {"1", "2", "3"}
    assert {row["seed"] for row in aggregate_rows} == {"1", "2", "3"}
    assert statistics_rows[0]["policy"] == "local"
    assert statistics_rows[0]["seed_count"] == "3"
    assert float(statistics_rows[0]["mean_latency_s_std"]) >= 0.0


def test_policies_share_task_ids_and_backend_state_resets(tmp_path):
    tasks = generate_synthetic_tasks(count=3, seed=21)
    policies = [
        BaselinePolicy("local", LocalOnlyPolicy(), EdgeExecutePolicy(), EDFCloudPolicy()),
        BaselinePolicy("edge", AlwaysEdgePolicy(), EdgeExecutePolicy(), EDFCloudPolicy()),
    ]
    first = run_baselines({21: tasks}, policies, output_dir=tmp_path / "first")
    second = run_baselines({21: tasks}, policies, output_dir=tmp_path / "second")

    with first[0].open(newline="", encoding="utf-8") as file:
        first_rows = list(csv.DictReader(file))
    with second[0].open(newline="", encoding="utf-8") as file:
        second_rows = list(csv.DictReader(file))
    task_ids_by_policy = {
        policy: {row["task_id"] for row in first_rows if row["policy"] == policy}
        for policy in ("local", "edge")
    }

    assert task_ids_by_policy["local"] == task_ids_by_policy["edge"]
    assert first_rows == second_rows


def test_edf_and_fifo_choose_different_cloud_execution_orders():
    tasks = [
        Task("first", "device-1", 0.0, 0.1, 1_000.0, 0.1, 100.0),
        Task("fifo-second", "device-1", 0.01, 0.1, 100.0, 0.1, 100.0),
        Task("edf-third", "device-1", 0.02, 0.1, 100.0, 0.1, 1.0),
    ]
    backend = SimPyBackend(config=SimulationConfig(cloud_cpu_capacity_mips=100.0))

    fifo_results = backend.run_episode(
        tasks,
        AlwaysEdgePolicy(),
        EdgeForwardPolicy(),
        FIFOCloudPolicy(),
        seed=1,
    )
    edf_results = backend.run_episode(
        tasks,
        AlwaysEdgePolicy(),
        EdgeForwardPolicy(),
        EDFCloudPolicy(),
        seed=1,
    )

    fifo_order = [
        result.task_id
        for result in sorted(fifo_results, key=lambda result: result.execution_start_time_s)
    ]
    edf_order = [
        result.task_id
        for result in sorted(edf_results, key=lambda result: result.execution_start_time_s)
    ]
    assert fifo_order[1:] == ["fifo-second", "edf-third"]
    assert edf_order[1:] == ["edf-third", "fifo-second"]
