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

Derived compute demand is estimated as:

`mean((instance_end - instance_start) × real_cpu_avg × reference_mips_per_core)`

over valid completed instances for the task. `instance_num` describes parallel
instances in a stage; summing their work and scheduling that result as one
serial task substantially overstates the modeled task demand. The adapter
therefore treats each accepted task as one representative instance. This
changes the workload semantics and is versioned as
`alibaba2017-mean-completed-instance-lognormal-payload-v2`; checkpoints made
with earlier task/payload adapters must not be reused. `real_cpu_avg` is documented as an
actual CPU number, not a fraction. The MIPS-per-core conversion is a
configurable simulation estimate, not an Alibaba measurement.

## Provenance table

| Target field | Trace source column | Direct or derived? | Mapping / unit | Missing-data policy |
|---|---|---|---|---|
| `arrival_time_s` | 2017 `batch_task.create_timestamp` | Derived | subtract trace origin, seconds | reject missing/invalid |
| `required_mi` | 2017 `batch_instance.start_timestamp`, `end_timestamp`, `real_cpu_avg` | Derived estimate | mean completed-instance runtime × actual CPU × configurable MIPS/core, MI | reject task if no valid completed instances |
| `input_size_mb` | None | Synthetic | deterministic lognormal; median 1 MB, sigma 1, capped at 50 MB | SHA-256 of task ID |
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

Verify it with the repository's official `SHA256SUM` and extract the needed
files into `data/raw/`. The command below **audits and reports** a local CSV; it
does not write a filtered CSV or a processed cache:

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
limit, the local CSVs audited on 2026-11-06 produced 2,891 selected tasks,
77,662 rejected task rows, and 147,849 rejected/incomplete auxiliary instance
records. The largest task rejection category was 77,487 multi-task jobs; these
are excluded because the independent-task simulator does not reconstruct their
DAGs. Counts are tied to the exact local CSVs and loader version; rerun the
audit after replacing either file.

For unit tests, use `pytest -q tests/test_trace_workloads.py`. The adapter
returns the same `Task` objects as `generate_synthetic_tasks()`. Chronological
splits are available with `split_trace_tasks()` and do not overlap. The RL
training and comparison scripts call this adapter directly; they re-read and
re-parse the raw CSVs on each run and do not currently write a cached canonical
dataset. Their `--max-trace-tasks` option limits accepted tasks retained, not
the number of source rows scanned: task/instance joins and source-file
fingerprints require reading the source files. For the current local audit,
use a limit above 2,891 to retain all accepted tasks. The `--format
2018-instance` option remains only for the legacy 2018 adapter.

The integrated RL scripts perform this loader/mapping step automatically when
invoked with `--source alibaba2017`. They parse the raw pair into canonical
`Task` objects in memory; no separate preprocessing command is required before
training. The current defaults (lognormal input, 0.1 MB output, 5-second
relative deadline, 1,000 reference MIPS, round-robin device assignment) are
assumptions supplied by `TraceWorkloadConfig`, not Alibaba measurements.

The 5-second default is intentionally preserved as a strict scenario, but it
is not calibrated to the observed compute demand. For an alternative
compute-relative scenario, the RL scripts accept
`--deadline-policy local-relative-2x`: each task's deadline is set to twice its
isolated execution time on its assigned device
(`2 × required_mi / device_cpu_mips`). This is a declared simulation SLA
assumption, not a deadline observed in Alibaba. Train and evaluate matching
checkpoints under each policy; do not compare a model trained with one
deadline basis against another. Report both cases because fixed 5 seconds
shows strict-SLA behavior while the relative case tests routing/queueing
decisions without making most trace tasks impossible by construction.

The optional `best-tier-calibrated` policy sets each deadline to
`max(1 second, 0.9 × required_mi / fastest_configured_MIPS + 0.5 seconds)`.
The multiplier was selected using the chronological training split only; with
the current accepted trace and topology it yields about 73% no-queue-feasible
training tasks. The suggested 3x multiplier was not used because the measured
no-queue feasibility was 100%, which removes infeasible tasks from the
comparison. This remains a synthetic SLA assumption, and the chronological
validation/test feasibility rates differ because the trace is not stationary.

The representative-instance work remains heavy-tailed. Recalculate
assigned-device service-time percentiles whenever the topology's device
assignment or CPU capacities change; the earlier p50/p95/p99 figures were
measured on a different topology.

**Important:** Alibaba Cluster Trace is an infrastructure workload trace, not a
complete MEC device dataset. Device assignment, battery, payload sizes,
deadlines, wireless conditions, and edge/cloud capacities are synthetic MEC
assumptions and must be reported as such.

## Multi-device topology integration

The contended RL regime compresses arrivals with `k=0.136067` and enables
seeded ±10% bandwidth and service-time variation. `scripts/calibrate_contention.py`
recomputes the offered load and baseline response from the local chronological
test split; the factor was selected to target 0.75 aggregate edge load against
12,500 MIPS.

Run the accepted 2017 tasks through the configured 24-device/three-edge topology:

```bash
python scripts/run_trace_workload.py \
  data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv \
  --source alibaba2017 --max-tasks 20 \
  --topology-config configs/alibaba-multi-topology.yaml
```

Accepted tasks are assigned deterministically in selection order and mapped
round-robin over three edges:

```text
task 0 -> device-1 -> edge-1
task 1 -> device-2 -> edge-2
task 2 -> device-3 -> edge-3
task 3 -> device-4 -> edge-1
...
```

This placement is synthetic; it is not present in Alibaba's records. Arrival
timestamps and derived MI are trace-derived. Payload sizes, deadlines, device
state, wireless links, Edge capacities, and Cloud capacity are simulation
assumptions. The runner exports per-task results and aggregate metrics. Raw
CSV files remain local and ignored by Git.
