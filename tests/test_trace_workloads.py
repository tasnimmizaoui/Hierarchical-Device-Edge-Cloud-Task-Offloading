import pytest

from mec_offloading.trace_workloads import (
    TraceWorkloadConfig,
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
