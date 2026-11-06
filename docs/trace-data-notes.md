# Workload trace: Alibaba Cluster Trace v2018

Official documentation: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md

## Dataset size and tables

The official v2018 documentation describes about 4,000 machines over eight days and six tables: `machine_meta`, `machine_usage`, `container_meta`, `container_usage`, `batch_instance`, and `batch_task`. The published combined archive is very large (documented around 98 GB in the separate-file listing), so do not download/extract everything for the first experiment.

## Verified schema mapping and data workflow

1. Read `trace_2018.md` and the linked schema first.
2. Decide whether the project's workload should represent independent batch tasks. The task `task_name` can encode DAG dependencies; the initial project scope can choose independent tasks, but document filtering/exclusion.
3. Use `batch_instance.csv` for the first adapter. Its verified fields are
   `instance_name`, `task_name`, `job_name`, `status`, `start_time`,
   `end_time`, and `cpu_avg`. Alibaba documents all time fields in seconds
   from the trace period origin and `cpu_avg` as average CPU usage where 100
   represents one core.
4. Download only the smallest required subset/table(s) using the official instructions, and record archive checksums.
5. Preprocess in chunks; retain a deterministic sampling seed and a compact selected window.
6. Map raw trace fields to `Task` fields in `docs/interfaces.md`; preserve source column names and mapping units.
7. Explicitly synthesize missing MEC-specific properties such as device battery, radio bandwidth/propagation, task deadline and edge association, unless another measured source is introduced.

The adapter uses the observed instance `start_time` as an arrival proxy because
the verified `batch_instance` schema has no task-creation timestamp. It derives
compute demand as:

`(end_time - start_time) × (cpu_avg / 100) × reference_mips_per_core`

The reference MIPS-per-core value is configurable and is not an Alibaba
measurement. Records with missing/invalid required fields, unaccepted status,
duplicate IDs, or DAG dependency suffixes are rejected and counted. The
default accepted statuses are `completed`, `finished`, `success`, `succeeded`,
and `terminated`; override them for a differently documented sample.

## Provenance table

| Target field | Trace source column | Direct or derived? | Mapping / unit | Missing-data policy |
|---|---|---|---|---|
| `arrival_time_s` | `batch_instance.start_time` | Derived | subtract minimum trace start, seconds | reject missing/invalid |
| `required_mi` | `end_time`, `start_time`, `cpu_avg` | Derived estimate | elapsed seconds × average core fraction × configurable MIPS/core, MI | reject invalid |
| `input_size_mb` | None | Synthetic | configured decimal MB | configuration default |
| `result_size_mb` | None | Synthetic | configured decimal MB | configuration default |
| `deadline_s` | None | Synthetic | configured relative seconds | configuration default |
| `device_id` | None | Synthetic assignment | deterministic round-robin configured IDs | configuration default |
| battery / radio state | None identified yet | Synthetic | configuration distribution | n/a |

## Commands

Raw data is not included in Git. Obtain the official dataset under its terms,
extract only a small `batch_instance.csv` sample into `data/raw/`, then run:

```bash
python scripts/prepare_trace_sample.py data/raw/batch_instance.csv
python scripts/run_trace_workload.py data/raw/batch_instance.csv --max-tasks 20
```

For unit tests, use `pytest -q tests/test_trace_workloads.py`. The adapter
returns the same `Task` objects as `generate_synthetic_tasks()`. Chronological
splits are available with `split_trace_tasks()` and do not overlap.

**Important:** Alibaba Cluster Trace is an infrastructure workload trace, not a complete MEC device dataset. Do not claim it directly provides battery, wireless conditions or per-task deadlines unless you verify those fields in a relevant schema.
