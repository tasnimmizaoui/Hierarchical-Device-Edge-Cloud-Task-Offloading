# Experiment protocol — v1

This protocol describes the current reproducible synthetic baseline experiment.
It is a development/evaluation protocol, not a claim that one policy is
universally superior.

## Reproduction

```bash
python scripts/run_baselines.py --seeds 1 2 3
```

Outputs are written to `results/dev-smoke/`:

- `task_results.csv`: detailed results with `policy` and `seed`;
- `aggregate_metrics.csv`: one aggregate row per policy and seed;
- `aggregate_statistics.csv`: mean and sample standard deviation across seeds.

Each seed generates one workload. That exact workload is passed to every policy
for that seed. Each policy run creates a fresh SimPy environment and fresh
resources. Random policies receive independent generators derived from the
experiment seed (`seed + 101`, `seed + 202`, and `seed + 303`).

## Scenario table

| Parameter | Value/distribution | Unit | Source/rationale |
|---|---|---|---|
| Number of devices | 1 | devices | Current backend scope |
| Number of edge servers | 1 | servers | Current backend scope |
| Cloud/HPC capacity | 20,000 | MIPS | `SimulationConfig` default |
| Device CPU | 1,000 | MIPS | Synthetic assumption |
| Edge CPU | 5,000 | MIPS | Synthetic assumption |
| Device-edge bandwidth/delay | 100 / 10 | Mbps / ms | Synthetic assumption |
| Edge-cloud bandwidth/delay | 1,000 / 40 | Mbps / ms | Synthetic assumption |
| Task arrivals | 0.05-second spacing | seconds | Seeded generator |
| Compute demand | Uniform(250, 1,000) | MI | Synthetic assumption |
| Input size | Uniform(0.5, 2.0) | MB | Synthetic assumption |
| Result size | Uniform(0.05, 0.2) | MB | Synthetic assumption |
| Relative deadline | Uniform(1, 4) | seconds | Synthetic assumption |
| Random seeds | User supplied, e.g. 1, 2, 3 | integers | Reproducibility |

The current backend does not support shared multi-device or multi-edge
resources. Therefore no multi-device result is reported in this phase. Adding
that topology is a separate backend extension, not an assumption hidden inside
the current CSVs.

## Baseline semantics

Placement and scheduling are separate:

| Label | Device decision | Edge decision | Cloud scheduler |
|---|---|---|---|
| `local` | local | unused | unused |
| `greedy_edge` | estimated lower latency | edge execution | unused |
| `edge_execute` | edge | edge execution | unused |
| `cloud_edf` | edge | forward cloud | earliest-deadline-first |
| `cloud_fifo` | edge | forward cloud | arrival order |
| `random` | seeded random | seeded random | seeded random rule |

`edge_execute` replaces the old ambiguous development label `edge_edf`: EDF
was never used when tasks stopped at the edge. `cloud_edf` means EDF is used
only after the edge forwards the task to the cloud.

## Metric definitions

- Mean, median and P95 latency use all returned task results.
- Deadline success/miss rates use all returned task results.
- Mean device energy uses returned tasks with no failure reason.
- Throughput is completed tasks divided by the latest completion timestamp.
- Communication bytes are the sum of input and result payload bytes crossing
  modeled links.
- Across-seed standard deviation is the sample standard deviation; it is zero
  when only one seed is present.

## Baseline audit findings

The three-seed development CSVs were audited against the policy constructors
and backend execution paths.

### Why `greedy_edge` and `edge_execute` match

`greedy_edge` means:

1. the device compares `required_mi / device_mips` with an estimated
   device-edge upload plus edge execution time;
2. it offloads when the edge estimate is lower;
3. the edge then uses `EdgeExecutePolicy`.

For the default synthetic scenario, the generated tasks all satisfy that
inequality, so `greedy_edge` selects `OFFLOAD_TO_EDGE` for every task. Its
edge policy is the same as `edge_execute`; therefore their routes,
latencies, energy, deadlines, and bytes are identical. This is a legitimate
equivalence for this scenario, not evidence that the greedy policy is
constant in general.

A direct probe confirms that the greedy policy changes decisions: a
small/fast-edge task is offloaded, while a task with a 1 Mbps link and a
100 MIPS edge is executed locally. A future placement experiment must include
both regimes. The smallest correction is to add a documented workload or
network-condition sweep; changing the policy is unnecessary.

### Why `cloud_edf` and `cloud_fifo` match

Both labels use the same device and edge decisions and differ only in the
cloud queue rule. In the default workload, the device-edge and edge-cloud
transfers space cloud arrivals sufficiently, and the cloud is fast enough
that queued jobs do not create a meaningful scheduling choice. Consequently
EDF and FIFO receive effectively one eligible job at dispatch, or jobs whose
arrival order and deadline order agree. Their current aggregate equality is
therefore expected.

A controlled test with cloud capacity reduced to 100 MIPS and deadlines
ordered differently from arrival order produces different execution orders:
FIFO preserves arrival order while EDF selects the earlier absolute deadline.
The smallest correction is to add a queue-contention scenario to the
experiment matrix; do not claim an EDF improvement from the current smoke
scenario.

The reproducible configuration is
`configs/cloud-contention.yaml`. With its validation workload, the measured
orders and outcomes were:

| Policy | Cloud execution order | Urgent task outcome |
|---|---|---|
| FIFO | `long-first`, `fifo-second`, `urgent-third` | deadline missed |
| EDF | `long-first`, `urgent-third`, `fifo-second` | deadline missed |

This is a valid distinction between scheduling policies, but it does not show
that EDF is better for every metric. The validation output reports simulation
timestamps and arrival-relative latency separately. The actual measurements
were:

| Policy | Task | Arrival | Execution start | Completion | Absolute deadline | Latency | Deadline |
|---|---|---:|---:|---:|---:|---:|---|
| FIFO | `long-first` | 0.0000 s | 0.0036 s | 10.0072 s | 100.0000 s | 10.0072 s | met |
| FIFO | `fifo-second` | 0.0500 s | 10.0036 s | 11.0072 s | 100.0500 s | 10.9572 s | met |
| FIFO | `urgent-third` | 0.0600 s | 11.0036 s | 12.0072 s | 0.2600 s | 11.9472 s | missed |
| EDF | `long-first` | 0.0000 s | 0.0036 s | 10.0072 s | 100.0000 s | 10.0072 s | met |
| EDF | `urgent-third` | 0.0600 s | 10.0036 s | 11.0072 s | 0.2600 s | 10.9472 s | missed |
| EDF | `fifo-second` | 0.0500 s | 11.0036 s | 12.0072 s | 100.0500 s | 11.9572 s | met |

Thus `11.9472 s` and `10.9472 s` are actual end-to-end latencies:
`completion_time_s - arrival_time_s`, not absolute completion timestamps. The
urgent task's relative deadline is `0.2 s` from its `0.06 s` arrival, giving an
absolute deadline of `0.26 s`; both policies miss it after the deliberately
long first task. Deadline success uses the same arrival-relative boundary:
`completion_time_s <= arrival_time_s + deadline_s`.

The reproducible greedy configuration is
`configs/greedy-switch.yaml`. It measured:

| Task | Route selected | End-to-end latency |
|---|---|---:|
| `local-candidate` | device | 1.0000 s |
| `edge-candidate` | device → edge | 0.3800 s |

The decisions come from the cost estimates: the large upload makes local
execution cheaper for the first task, while the small upload makes edge
execution cheaper for the second. No task identifier is used by the policy.
The same file also evaluates explicit conditions:

- `slow-link`: 10 Mbps device-edge link, 5,000 MIPS edge;
- `fast-link`: 1,000 Mbps device-edge link, 5,000 MIPS edge;
- `slow-edge`: 100 Mbps device-edge link, 100 MIPS edge.

The measured routes were respectively `device/edge`, `edge/edge`, and
`device/device` for the two workload tasks. This confirms that both network
bandwidth and edge capacity can change the greedy decision.

### Meaning of the six labels

All six labels are technically meaningful, but they test different layers:

- `local` is the local-execution reference;
- `greedy_edge` tests adaptive device placement;
- `edge_execute` isolates fixed edge placement/execution;
- `cloud_edf` tests cloud forwarding plus EDF scheduling;
- `cloud_fifo` is the matching cloud-forwarding control with FIFO;
- `random` tests seeded stochastic decisions at all three levels.

`greedy_edge` and `edge_execute` are redundant only for the current workload,
not by definition. `cloud_edf` and `cloud_fifo` are necessary as a matched
scheduling comparison, but require a deliberately congested scenario to
produce different outcomes.

### Path and metric consistency

The CSV routes match the configured paths:

- local: `device`, zero communication bytes and zero communication delay;
- edge: `device->edge`, one upload and one result download;
- cloud: `device->edge->cloud`, device-edge and edge-cloud transfers in both
  directions, hence more bytes and delay than the edge route.

Device energy includes local CPU energy or the device radio's upload/download
energy. Edge-cloud infrastructure energy is not included in device energy.
This explains why cloud and edge routes can have equal device energy while
cloud communication bytes and delay are higher.

The detailed `execution_start_time_s` field records the actual timestamp after
the relevant SimPy compute resource is acquired. Therefore it includes queue
waiting for local, edge, and cloud execution. It is separate from task arrival
and completion time, and it does not change latency, energy, routes, or
communication totals.

## Per-run metadata

Save policy, scenario ID, seed, config snapshot, git commit, package versions, start/end timestamps and result file paths.

## Analysis checklist

- [ ] Count generated, completed, failed and unfinished tasks.
- [ ] State whether latency averages include only completed tasks; report failure rate separately.
- [ ] Report latency distribution, not only the mean.
- [ ] Define energy accounting and denominator.
- [ ] Define communication overhead (messages, bytes, simulated delay, or all).
- [x] Use multiple seeds and variability/error bars.
- [ ] Explain unexpected outcomes and cases where hierarchical RL loses.
