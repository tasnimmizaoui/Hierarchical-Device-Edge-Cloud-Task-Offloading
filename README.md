# Hierarchical Device–Edge–Cloud Task Offloading

A reproducible SimPy simulator and evaluation
pipeline for hierarchical task offloading across devices, Edge servers, and a
shared Cloud/HPC scheduler.

The repository contains:

- a discrete-event MEC backend with local, Edge, and Cloud execution paths;
- configurable multi-device/multi-Edge topologies;
- synthetic workloads for fast development;
- an adapter for Alibaba Cluster Trace 2017;
- deterministic Random, Greedy, EDF, FIFO, and related baselines;
- integrated Device, Edge, and Cloud DQN policy adapters;
- CSV exports, validation scripts, and experiment metadata.

The simulator reports end-to-end latency, queue waiting, deadline status,
device energy, communication delay/bytes, execution start time, and source
device/Edge identifiers.

## Current scope and limitations

The canonical simulator is [SimPy](https://simpy.readthedocs.io/). The
repository implements the MEC model explicitly so Device, Edge, and Cloud
policy boundaries remain testable and independent.

The backend supports multiple devices and Edge servers feeding one shared
Cloud worker. Each device and Edge currently has one modeled CPU resource;
multi-core devices and multiple concurrent Cloud workers are not modeled.

Alibaba Cluster Trace is a datacenter workload, not a mobile-MEC trace. Its
timestamps and compute evidence are adapted into the simulator. Device
assignment, payload sizes, deadlines, batteries, wireless links, and MEC
capacities are explicit simulation assumptions. See
[`docs/trace-data-notes.md`](docs/trace-data-notes.md).

The integrated RL workflow uses the same SimPy backend and policy protocols.
It supports synthetic and Alibaba 2017 workloads, configurable topology
variants, held-out splits, feedback modes, and DQN/Double-DQN checkpoints.
The transition formulation is currently a contextual one-step decision with
terminal or next-decision credit; it is not a full multi-step Gymnasium
environment.

## Installation

Python 3.11 is the reference runtime.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,rl,web]"
```

On Windows PowerShell, replace the activation command with:

```powershell
py -3.11 -m venv .venv311
.venv311\Scripts\Activate.ps1
```

Check the environment:

```bash
python scripts/check_environment.py
```

All commands below are run from the repository root. If the package is not
installed, prefix commands with `PYTHONPATH=src` or use the virtual
environment's Python executable.

## Verify the repository

Run the complete test suite:

```bash
python -m pytest -q
```

Run Ruff:

```bash
ruff check .
```

Run the small route smoke test:

```bash
python scripts/run_simulation_smoke.py
```

The smoke test is a sanity check, not a scientific comparison.

## Synthetic workloads and baselines

Run the reproducible baseline comparison on identical seeded workloads:

```bash
python scripts/run_baselines.py \
  --seeds 1 2 3 \
  --task-count 10 \
  --output-dir results/dev-smoke
```

The runner writes:

```text
results/dev-smoke/task_results.csv
results/dev-smoke/aggregate_metrics.csv
results/dev-smoke/aggregate_statistics.csv
```

The baseline labels are defined in
[`src/mec_offloading/evaluation/README.md`](src/mec_offloading/evaluation/README.md)
and [`docs/experiment_protocol.md`](docs/experiment_protocol.md). Every policy
receives the same workload for a given seed; random policies receive
independent, deterministic random streams.

Run the controlled policy-validation scenarios:

```bash
python scripts/run_scenario_validation.py configs/greedy-switch.yaml
python scripts/run_scenario_validation.py configs/cloud-contention.yaml
```

Demonstrate the three-device/two-Edge/shared-Cloud topology:

```bash
python scripts/run_multi_topology_demo.py
```

This prints actual SimPy results, including source IDs, routes, latency,
energy, deadlines, and Cloud scheduling order.

## Alibaba Cluster Trace 2017

Raw trace files are not committed. Obtain the official Alibaba 2017 archive
under its terms and place the required files at:

```text
data/raw/batch_task.csv
data/raw/batch_instance.csv
```

Validate the complete local input:

```bash
python scripts/prepare_trace_sample.py \
  data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv \
  --format 2017-task
```

Run a small trace workload on the versioned topology:

```bash
python scripts/run_trace_workload.py \
  data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv \
  --source alibaba2017 \
  --max-tasks 20 \
  --topology-config configs/alibaba-multi-topology.yaml \
  --output-dir results/trace-multi-topology
```

Accepted tasks are assigned deterministically in selection order:

```text
task 0 -> device-1 -> edge-1
task 1 -> device-2 -> edge-1
task 2 -> device-3 -> edge-2
```

The runner writes `task_results.csv` and `aggregate_metrics.csv` under the
selected output directory. It reports selected records, rejection reasons,
source IDs, routes, latency, energy, and deadline outcomes.

The same runner can exercise the preserved synthetic generator on the same
topology:

```bash
python scripts/run_trace_workload.py \
  --source synthetic \
  --max-tasks 6 \
  --topology-config configs/alibaba-multi-topology.yaml
```

The trace adapter and provenance rules are documented in
[`docs/trace-data-notes.md`](docs/trace-data-notes.md). Raw data and generated
results are ignored by Git.

## Integrated RL experiments

Train on the default synthetic scenarios:

```bash
python scripts/train_integrated_rl.py \
  --source synthetic \
  --episodes 100 \
  --tasks 100 \
  --seed 42 \
  --output artifacts/models/integrated
```

Train on Alibaba 2017 with the configured multi-device topology:

```bash
python scripts/train_integrated_rl.py \
  --source alibaba2017 \
  --episodes 100 \
  --tasks 100 \
  --max-trace-tasks 5000 \
  --topology-config configs/alibaba-multi-topology.yaml \
  --topology-augmentation sweep \
  --deadline-policy best-tier-calibrated \
  --feedback periodic \
  --seed 42 \
  --output artifacts/models/integrated
```

Training writes model checkpoints, `metadata.json`, and
`training_log.csv` under the selected output directory. Training and validation
seeds are kept disjoint by the script.

Evaluate trained checkpoints against baselines:

```bash
python scripts/compare_integrated_policies.py \
  --source alibaba2017 \
  --trace-path data/raw/batch_task.csv \
  --trace-instances data/raw/batch_instance.csv \
  --topology-config configs/alibaba-multi-topology.yaml \
  --models artifacts/models/integrated \
  --seeds 10001 10002 10003 \
  --tasks 5000 \
  --max-trace-tasks 5000 \
  --deadline-policy best-tier-calibrated \
  --feedback periodic \
  --output results/integrated-comparison
```

Evaluation writes `task_results.csv`, `aggregate_metrics.csv`,
`aggregate_statistics.csv`, and `metadata.json`. The comparator verifies model
schema metadata and rejects overlapping training, validation, and evaluation
seeds.

For the full multi-seed feedback study, use
[`scripts/run_feedback_ablation.py`](scripts/run_feedback_ablation.py). It
launches the training and comparison commands with matching trace, topology,
algorithm, feedback, and seed settings. This is a long-running campaign; use
smaller episode/task counts for a smoke run first.

## Repository map

```text
src/mec_offloading/domain/       Shared tasks, states, actions, metrics, ports
src/mec_offloading/simulation/   SimPy backend and topology configuration
src/mec_offloading/workloads.py  Synthetic workload generator
src/mec_offloading/trace_workloads.py
                                  Alibaba adapters and deterministic assignment
src/mec_offloading/baselines/    Interpretable policy implementations
src/mec_offloading/rl/           DQN agents, observations, credit, scenarios
src/mec_offloading/evaluation/   CSV export and aggregate metrics
configs/                          Versioned scenario/topology parameters
scripts/                          Reproducible validation and experiment entry points
tests/                            Unit and integration tests
docs/                             Architecture, interfaces, protocol, provenance
report/                           IEEE article sources and reproducible report assets
```

Start with:

1. [`docs/architecture.md`](docs/architecture.md)
2. [`docs/interfaces.md`](docs/interfaces.md)
3. [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
4. [`docs/trace-data-notes.md`](docs/trace-data-notes.md)
5. [`docs/rl_experiment_report.md`](docs/rl_experiment_report.md)

Generated results, model binaries, figures, and raw datasets are intentionally
excluded from normal Git commits. Record the command, configuration, seed set,
package environment, and source commit for every result used in the report.
