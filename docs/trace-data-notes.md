# Workload trace: Alibaba Cluster Trace v2018

Official documentation: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md

## Dataset size and tables

The official v2018 documentation describes about 4,000 machines over eight days and six tables: `machine_meta`, `machine_usage`, `container_meta`, `container_usage`, `batch_instance`, and `batch_task`. The published combined archive is very large (documented around 98 GB in the separate-file listing), so do not download/extract everything for the first experiment.

## Recommended data workflow

1. Read `trace_2018.md` and the linked schema first.
2. Decide whether the project's workload should represent independent batch tasks. The task `task_name` can encode DAG dependencies; the initial project scope can choose independent tasks, but document filtering/exclusion.
3. Identify the exact fields needed for arrival time and compute demand from the schema. Do not infer column meanings from their names alone.
4. Download only the smallest required subset/table(s) using the official instructions, and record archive checksums.
5. Preprocess in chunks; retain a deterministic sampling seed and a compact selected window.
6. Map raw trace fields to `Task` fields in `docs/interfaces.md`; preserve source column names and mapping units.
7. Explicitly synthesize missing MEC-specific properties such as device battery, radio bandwidth/propagation, task deadline and edge association, unless another measured source is introduced.

## Provenance table to fill

| Target field | Trace source column | Direct or derived? | Mapping / unit | Missing-data policy |
|---|---|---|---|---|
| `arrival_time_s` | TBD | TBD | seconds relative to trace start | TBD |
| `required_mi` | TBD | TBD | convert/estimate to million instructions | TBD |
| `input_size_mb` | TBD | TBD | MB | synthetic if unavailable |
| `result_size_mb` | TBD | TBD | MB | synthetic if unavailable |
| `deadline_s` | None identified yet | Synthetic | documented distribution/scenario rule | n/a |
| `device_id` | TBD | Derived/synthetic | assign task stream to simulated device | TBD |
| battery / radio state | None identified yet | Synthetic | configuration distribution | n/a |

**Important:** Alibaba Cluster Trace is an infrastructure workload trace, not a complete MEC device dataset. Do not claim it directly provides battery, wireless conditions or per-task deadlines unless you verify those fields in a relevant schema.
