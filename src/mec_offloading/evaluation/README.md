# Shared — evaluation

The evaluation layer runs policy/scenario/seed combinations, writes detailed
CSV results, and aggregates metrics. Training seeds must remain separate from
held-out evaluation seeds.

`runner.py` provides the reproducible baseline runner. It uses one generated
task list per seed and reuses that exact list for every policy. A policy builder
creates fresh policy objects per seed, which is important for Random policy
state. It writes:

- `task_results.csv`: one row per task and policy;
- `aggregate_metrics.csv`: one row per policy and seed;
- `aggregate_statistics.csv`: across-seed mean and sample standard deviation.

The deadline rates use all returned task results as the denominator. Throughput
is completed tasks divided by the latest completion timestamp. These definitions
are intentionally explicit so later RL comparisons use the same formulas.

Baseline labels mean:

- `local`: every task executes on the device;
- `greedy_edge`: device estimates local versus device-to-edge execution;
- `edge_execute`: every offloaded task executes at the edge; cloud EDF is unused;
- `cloud_edf`: every task is forwarded to cloud and the cloud queue uses EDF;
- `cloud_fifo`: every task is forwarded to cloud and the cloud queue uses FIFO;
- `random`: device placement, edge forwarding, and cloud rule are independently
  sampled from seeded random generators.

The current backend intentionally models one device, one edge server, and one
cloud node. The workload and runner are seed-aware, but a multi-device,
multi-edge comparison must wait for a topology-capable backend extension;
current results must not be described as multi-device results.

Run the current baseline experiment from the repository root:

```bash
python scripts/run_baselines.py --seeds 1 2 3
```

The generated `results/` directory is ignored by Git. Keep the command,
seeds, configuration, and package versions with any result used in the report.
