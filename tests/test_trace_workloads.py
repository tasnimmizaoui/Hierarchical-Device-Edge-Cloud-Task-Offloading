import pytest

from mec_offloading.trace_workloads import (
    TraceWorkloadConfig,
    load_alibaba_2017_batch_tasks,
    load_alibaba_batch_instances,
    load_workload,
    split_trace_tasks,
)

HEADER = (
    "instance_name,task_name,job_name,task_type,status,start_time,end_time,"
    "machine_id,seq_no,total_seq_no,cpu_avg,cpu_max,mem_avg,mem_max\n"
)


def write_fixture(tmp_path, body: str):
    path = tmp_path / "batch_instance.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_trace_parsing_maps_relative_time_and_configured_mi(tmp_path):
    path = write_fixture(
        tmp_path,
        "i-1,random,job-1,Type,Terminated,10,12,m-1,1,1,50,60,2,3\n",
    )
    result = load_alibaba_batch_instances(
        path,
        config=TraceWorkloadConfig(
            reference_mips_per_core=2_000,
            input_size_mb=3,
            result_size_mb=0.5,
            deadline_s=8,
        ),
    )
    task = result.tasks[0]
    assert task.arrival_time_s == 0
    assert task.required_mi == 2_000
    assert task.input_size_mb == 3
    assert task.result_size_mb == 0.5
    assert task.task_id == "job-1/random/i-1"


def test_trace_rejects_missing_invalid_failed_and_dag_rows(tmp_path):
    path = write_fixture(
        tmp_path,
        "ok,random,job-1,T,success,10,11,m,1,1,50,50,1,1\n"
        "bad,random,job-2,T,failed,10,11,m,1,1,50,50,1,1\n"
        "missing,random,job-3,T,success,,11,m,1,1,50,50,1,1\n"
        "dag,M2_1,job-4,T,success,10,11,m,1,1,50,50,1,1\n",
    )
    result = load_alibaba_batch_instances(path)
    assert result.selected_count == 1
    assert result.rejected_count == 3


def test_trace_ids_splits_and_source_selection_are_stable(tmp_path):
    path = write_fixture(
        tmp_path,
        "".join(
            f"i-{i},random,job-{i},T,success,{i},{i + 1},m,1,1,50,50,1,1\n"
            for i in range(5)
        ),
    )
    first = load_alibaba_batch_instances(path)
    second = load_alibaba_batch_instances(path)
    assert first.tasks == second.tasks
    splits = split_trace_tasks(first.tasks)
    assert [task.task_id for task in splits["train"]] == [
        "job-0/random/i-0",
        "job-1/random/i-1",
        "job-2/random/i-2",
    ]
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert load_workload("alibaba", trace_path=path) == list(first.tasks)
    assert len(load_workload("synthetic", count=2, seed=4)) == 2


def test_trace_configuration_validates_reference_values():
    with pytest.raises(ValueError):
        TraceWorkloadConfig(reference_mips_per_core=0)


def test_max_tasks_stops_without_counting_remaining_rows_as_rejected(tmp_path):
    path = write_fixture(
        tmp_path,
        "".join(
            f"i-{i},random,job-{i},T,Terminated,{i},{i + 1},m,1,1,50,50,1,1\n"
            for i in range(3)
        ),
    )
    result = load_alibaba_batch_instances(
        path, config=TraceWorkloadConfig(max_tasks=1)
    )
    assert result.selected_count == 1
    assert result.rejected_count == 0
    assert result.stopped_early is True


def test_alibaba_2017_task_loader_uses_creation_time_and_requested_cpu(tmp_path):
    task_path = tmp_path / "batch_task.csv"
    task_path.write_text(
        "10,20,job-1,task-1,2,Terminated,3,4\n"
        "30,40,job-2,task-1,1,Failed,2,4\n",
        encoding="utf-8",
    )
    instance_path = tmp_path / "batch_instance.csv"
    instance_path.write_text(
        "12,14,job-1,task-1,1,Terminated,1,1,2,1.5,1,1\n",
        encoding="utf-8",
    )
    result = load_alibaba_2017_batch_tasks(
        task_path,
        instance_path=instance_path,
        config=TraceWorkloadConfig(reference_mips_per_core=1_000),
    )
    assert result.selected_count == 1
    assert result.rejected_count == 1
    assert result.auxiliary_rejected_count == 0
    task = result.tasks[0]
    assert task.arrival_time_s == 10
    assert task.required_mi == 3_000
    assert task.task_id == "job-job-1/task-task-1"
    assert load_workload(
        "alibaba2017",
        trace_path=task_path,
        trace_instance_path=instance_path,
    ) == [task]


def test_alibaba_2017_rejects_multi_task_jobs_to_avoid_unknown_dags(tmp_path):
    task_path = tmp_path / "batch_task.csv"
    task_path.write_text(
        "10,20,job-1,task-1,1,Terminated,1,1\n"
        "11,21,job-1,task-2,1,Terminated,1,1\n",
        encoding="utf-8",
    )
    instance_path = tmp_path / "batch_instance.csv"
    instance_path.write_text(
        "12,14,job-1,task-1,1,Terminated,1,1,1,1,1,1\n"
        "13,15,job-1,task-2,1,Terminated,1,1,1,1,1,1\n",
        encoding="utf-8",
    )
    result = load_alibaba_2017_batch_tasks(task_path, instance_path=instance_path)
    assert result.selected_count == 0
    assert result.rejected_count == 2
