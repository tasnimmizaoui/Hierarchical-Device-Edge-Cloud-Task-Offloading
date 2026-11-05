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
