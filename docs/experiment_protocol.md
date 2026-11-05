# Experiment protocol — v1 template

Freeze these before final comparison runs:

- Simulation engine, version and commit.
- Python/dependency versions.
- Device/edge/cloud counts and compute capacities.
- Workload source, selected subset/window and mapping into tasks.
- Synthetic distributions for deadlines, payloads, battery, bandwidth and propagation.
- Queue disciplines, resource concurrency and routing model.
- Reward equation, weights and observation normalization.
- Exact placement/scheduling semantics for all policies.
- Seeds, repeats, episode/run length, warm-up and drain behavior.
- Treatment of tasks left queued at simulation end.

## Scenario table

| Parameter | Value/distribution | Unit | Source/rationale |
|---|---|---|---|
| Number of devices | TBD | devices | |
| Number of edge servers | TBD | servers | |
| Cloud/HPC capacity | TBD | MIPS/cores | |
| Device CPU | TBD | MIPS | |
| Edge CPU | TBD | MIPS | |
| Device-edge bandwidth/delay | TBD | Mbps/ms | |
| Edge-cloud bandwidth/delay | TBD | Mbps/ms | |
| Task arrival process | TBD | tasks/s | |
| Task compute demand | TBD | MI | |
| Input/output size | TBD | MB | |
| Relative deadline | TBD | s | |
| Initial battery | TBD | J or fraction | |
| Random seeds | TBD | integers | |

## Per-run metadata

Save policy, scenario ID, seed, config snapshot, git commit, package versions, start/end timestamps and result file paths.

## Analysis checklist

- [ ] Count generated, completed, failed and unfinished tasks.
- [ ] State whether latency averages include only completed tasks; report failure rate separately.
- [ ] Report latency distribution, not only the mean.
- [ ] Define energy accounting and denominator.
- [ ] Define communication overhead (messages, bytes, simulated delay, or all).
- [ ] Use multiple seeds and variability/error bars.
- [ ] Explain unexpected outcomes and cases where hierarchical RL loses.
