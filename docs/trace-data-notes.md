# Workload trace: Alibaba Cluster Trace 2017

Official documentation:
[Alibaba 2017](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md)
and [Alibaba 2018](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)

## Recommendation: Alibaba Cluster Trace 2017

Use the 2017 release for the project workload. Its official `batch_task.csv`
contains `create_timestamp`, which is a genuine task creation/release time,
and `plan_cpu`, which is a per-instance CPU request. Its matching
`batch_instance.csv` contains actual instance start/end timestamps and
`real_cpu_avg`, allowing a derived computation estimate without calling an
execution start an arrival.

The 2017 release covers a 24-hour production-cluster period and is much more
practical for the Monday deadline than 2018. It is also headerless and requires
joining two files, which the adapter now handles.

## Dataset comparison

| Candidate | Arrival evidence | Compute evidence | DAG/retry implications | Practical size/effort |
|---|---|---|---|---|
| 2017 | `batch_task.create_timestamp` | `batch_task.plan_cpu` plus successful `batch_instance` duration and `real_cpu_avg` | Task dependencies are not exposed as a direct field; the loader accepts only single-task jobs. Instance rows represent retries via `seq_no`; failed/incomplete attempts are excluded and completed attempts are aggregated. | Official archive `trace_201708.tgz`; about 1.3k machines and a 24-hour period. Two headerless CSVs must be joined. |
| 2018 | `batch_instance.start_time` only | `cpu_avg` and instance duration | `task_name` encodes DAG dependencies; `batch_instance` has no creation time. | Official documentation lists a 48 GB archive, 280 GB extracted, or a 98 GB separate-file listing. Not realistic to process fully before Monday. |

2018 execution `start_time` is not original task arrival. It is when an
instance began running after scheduling. Using it as arrival would erase
queueing and release-time behavior. The current legacy 2018 adapter retains
this as an explicitly named arrival proxy only.

## Verified 2017 schema and mapping

The 2017 files are headerless. `batch_task.csv` fields are
`create_timestamp`, `modify_timestamp`, `job_id`, `task_id`, `instance_num`,
`status`, `plan_cpu`, and `plan_mem`. `batch_instance.csv` fields include
`start_timestamp`, `end_timestamp`, `job_id`, `task_id`, `machineID`, `status`,
`seq_no`, `total_seq_no`, and `real_cpu_avg`. Official timestamps are seconds
relative to the trace start. The adapter accepts completed/terminated records,
rejects invalid or incomplete records, and reports rejection counts.

Derived compute demand is:

`sum((instance_end - instance_start) × real_cpu_avg × reference_mips_per_core)`

over valid terminated instances for the task. `real_cpu_avg` is documented as
an actual CPU number, not a fraction. The MIPS-per-core conversion is a
configurable simulation estimate, not an Alibaba measurement.

## Provenance table

| Target field | Trace source column | Direct or derived? | Mapping / unit | Missing-data policy |
|---|---|---|---|---|
| `arrival_time_s` | 2017 `batch_task.create_timestamp` | Derived | subtract trace origin, seconds | reject missing/invalid |
| `required_mi` | 2017 `batch_instance.start_timestamp`, `end_timestamp`, `real_cpu_avg` | Derived estimate | sum runtime × actual CPU × configurable MIPS/core, MI | reject invalid/incomplete instances |
| `input_size_mb` | None | Synthetic | configured decimal MB | configuration default |
| `result_size_mb` | None | Synthetic | configured decimal MB | configuration default |
| `deadline_s` | None | Synthetic | configured relative seconds | configuration default |
| `device_id` | None | Synthetic assignment | deterministic round-robin configured IDs | configuration default |
| battery / radio state | None identified yet | Synthetic | configuration distribution | n/a |

## Commands

Raw data is not included in Git. Obtain the official 2017 archive under its
terms from the link in `README2017.md`:

```text
http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2017Traces/alibaba-trace-2017.tar.gz
```

Verify it with the repository's official `SHA256SUM`, extract only the needed
files, and place a manageable filtered sample in `data/raw/`:

```bash
python scripts/prepare_trace_sample.py data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv
python scripts/run_trace_workload.py data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv --max-tasks 20
```

The audit command reports `selected`, `task_rejected`, and
`instance_records_rejected` separately. With `--max-tasks 20`, it stops after
20 usable tasks and reports `stopped_early=True`; rejection counts then cover
records examined before that limit, not the entire source file. Without a
limit, the current real sample produced 381 selected tasks, 80,172 rejected
task rows, and 46,536 rejected/incomplete auxiliary instance records. The
largest task rejection category was 77,487 multi-task jobs; these are excluded
because the independent-task simulator does not reconstruct their DAGs.

For unit tests, use `pytest -q tests/test_trace_workloads.py`. The adapter
returns the same `Task` objects as `generate_synthetic_tasks()`. Chronological
splits are available with `split_trace_tasks()` and do not overlap. The
`--format 2018-instance` option remains only for the legacy 2018 adapter.

**Important:** Alibaba Cluster Trace is an infrastructure workload trace, not a
complete MEC device dataset. Device assignment, battery, payload sizes,
deadlines, wireless conditions, and edge/cloud capacities are synthetic MEC
assumptions and must be reported as such.

## Multi-device topology integration

Run the accepted 2017 tasks through the configured three-device topology:

```bash
python scripts/run_trace_workload.py \
  data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv \
  --source alibaba2017 --max-tasks 20 \
  --topology-config configs/alibaba-multi-topology.yaml
```

Accepted tasks are assigned deterministically in selection order:

```text
task 0 -> device-1 -> edge-1
task 1 -> device-2 -> edge-1
task 2 -> device-3 -> edge-2
task 3 -> device-1 -> edge-1
...
```

This placement is synthetic; it is not present in Alibaba's records. Arrival
timestamps and derived MI are trace-derived. Payload sizes, deadlines, device
state, wireless links, Edge capacities, and Cloud capacity are simulation
assumptions. The runner exports per-task results and aggregate metrics. Raw
CSV files remain local and ignored by Git.
