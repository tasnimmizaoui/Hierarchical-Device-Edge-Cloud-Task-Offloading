# Hierarchical MEC Task Offloading — Project Repository

**INSAT · Futurs Réseaux Mobiles · 2026–2027**  
**Team:** Student 1 — Simulation & Evaluation · Student 2 — RL & Coordination

## Project goal

Implement and evaluate a three-level task-offloading architecture:

- **Device:** execute locally or offload to a nearby edge.
- **Edge:** execute locally or forward a task to Cloud/HPC.
- **Cloud/HPC:** schedule tasks received from edge nodes.

The mandatory evaluation compares Random, Greedy/Nearest, EDF, Centralized RL and Hierarchical RL using end-to-end latency, device energy, deadline satisfaction and coordination overhead.


## 🧠 Intégration RL V2 — Étudiante 2

Cette version fusionne le code Double-DQN de `RL.zip` dans le dépôt principal :
le **moteur SimPy unique** exécute désormais les tâches pour les agents
Device/Edge/Cloud, l'agent centralisé et les heuristiques. Un visualiseur web
lit les résultats CSV communs. La V2 ajoute des récompenses terminales par
tâche, un feedback causal différé, des checkpoints choisis sur validation, des
ablation de fréquence et le suivi de l'apprentissage. Voir le
[guide technique et limites de la V2](docs/ETUDIANTE_2_V2_FR.md) et
[le guide initial de fusion](docs/INTEGRATION_RL.md). Le
[rapport expérimental](docs/rl_experiment_report.md) détaille les résultats
historiques v8/v9 et le nouveau calibrage contendu; les scores historiques ne
sont pas comparables aux réentraînements à venir.

```bash
pip install -e ".[dev,rl,web]"
python scripts/run_feedback_ablation.py --seeds 42 43 44 \
  --episodes 12 --tasks 100 --max-trace-tasks 5000
streamlit run web_app.py
```

Le mode par défaut utilise des tâches synthétiques et la topologie de
développement 1/1/1. Pour le jeu Alibaba local et la topologie configurée
(24 devices, 3 edges, cloud partagé), utiliser le même trace et la même limite
pour entraînement et test. Le mode de charge contendu calibré est `k=0.136067` :

```bash
python scripts/train_integrated_rl.py --source alibaba2017 --episodes 18 \
  --tasks 100 --max-trace-tasks 5000 --topology-augmentation sweep \
  --arrival-compression 0.136067 --seed 42
python scripts/compare_integrated_policies.py --source alibaba2017 \
  --feedback periodic --seeds 10001 10002 10003 --tasks 5000 \
  --max-trace-tasks 5000 --models artifacts/models/integrated
```

Le loader joint et transforme les CSV en tâches, puis crée des splits
chronologiques train/validation/test. Les données préparées ne sont pas
persistées : elles sont reparsées à chaque commande. Alibaba ne fournit pas
tous les champs MEC ; les payloads, deadlines et affectations sont des
hypothèses, documentées dans
[`docs/ETUDIANTE_2_V2_FR.md`](docs/ETUDIANTE_2_V2_FR.md).

Le protocole actuel entraîne sur neuf des 13 variantes et garde quatre
topologies (dont `combined_stress`) pour le test final. Le dashboard conserve
les résultats v8/v9 comme référence historique. La méthode et les limites sont dans
[`docs/INTEGRATION_RL.md`](docs/INTEGRATION_RL.md).

**Limite de capacité :** le SimPy backend supporte plusieurs devices et edges,
mais un seul slot CPU par device/edge et un worker Cloud. Il n'est pas encore
multi-core/multi-worker. Le DQN intégré donne un crédit terminal par tâche ;
il s'agit d'une formulation contextuelle à un pas, pas d'un environnement
Gymnasium multi-étapes. Réentraîner les modèles après tout changement de schéma
d'observation ou de topologie.

## Scope rule

Complete the mandatory three-agent system, baselines and reproducible experiments before attempting federated learning. The bonus extension is not part of the initial critical path.

## Team ownership

| Student | Owns | Main outputs |
|---|---|---|
| **Student 1 — Simulation & Evaluation** | Simulator feasibility and implementation, workload adapter, topology, compute/network queues, energy and latency model, Random/Greedy/EDF baselines, experiment runner and metric validation | Working simulator, trace/synthetic workload pipeline, baseline policies, metric tests, reproducible CSV runs |
| **Student 2 — RL & Coordination** | Device/edge/cloud agents, reward design, training, feedback protocol and centralized RL baseline | Three policy agents, model checkpoints, training configs/logs, centralized RL, coordination variants |
| **Both** | Freeze interfaces, integration, end-to-end tests, experiment protocol, result analysis, IEEE article, oral presentation and demo | Integrated system and complete submission |

Ownership is primary responsibility, not exclusive knowledge. Both students must be able to explain and reproduce the complete system.

## Technology choices

- **Python 3.11** for a consistent development environment.
- **SimPy 4.x:** the selected discrete-event kernel. We implement the small
  MEC model ourselves so device, edge, and cloud decisions remain separate.
  The decision and source-code evidence are in
  [`docs/decisions/0001-simulator-selection.md`](docs/decisions/0001-simulator-selection.md).
- Gymnasium is an optional dependency for environment adapters; the current
  integrated Double-DQN workflow calls the shared policy interfaces directly
  and is not a multi-step `gymnasium.Env`.
- The current Double-DQN implementation uses the project's PyTorch agent, not
  Stable-Baselines3.
- **NumPy, Pandas, Matplotlib, PyYAML** for computation, data and analysis.
- **pytest + Ruff + GitHub Actions** for tests, linting and reproducibility checks.

Do not add a second simulator. Student 2's RL code must use the policy
interfaces in `src/mec_offloading/domain/ports.py`.

## Team workflow

The repository is already initialized. Each student should:

1. Pull the latest `main` branch.
2. Create a short-lived branch for one task, for example
   `student-2/device-policy-adapter`.
3. Run the tests before editing.
4. Make a focused change and add or update tests.
5. Run `pytest -q` and `ruff check .`.
6. Open a pull request and request review from the other student.

Keep `main` stable. Do not change the shared task fields, units, action
meanings, or policy method signatures without reviewing
[`docs/interfaces.md`](docs/interfaces.md) together.

## Local setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,rl]"
python scripts/check_environment.py
pytest
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, or call `.venv\Scripts\python.exe` directly.

### Linux / macOS / WSL

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,rl]"
python scripts/check_environment.py
pytest
```

For Student 1's initial simulator spike, the non-RL dependencies are sufficient:

```bash
pip install -e ".[dev]"
```

For Student 2's separate mock-environment work, install the RL extra:

```bash
pip install -e ".[dev,rl]"
```

Start with CPU training. A GPU is not required for the small DQN smoke test.

## Start here

After installation, run the complete verification sequence:

```bash
python scripts/check_environment.py
pytest -q
python scripts/run_simulation_smoke.py
python scripts/run_baselines.py --seeds 1 2 3
ruff check .
```

Read the documentation in this order:

1. [`docs/architecture.md`](docs/architecture.md) — task lifecycle and
   decision boundaries.
2. [`docs/interfaces.md`](docs/interfaces.md) — shared data, units, actions,
   observations, and Python protocols.
3. [`docs/experiment_protocol.md`](docs/experiment_protocol.md) — baselines,
   seeds, metrics, and current limitations.
4. [`docs/task-board.md`](docs/task-board.md) — remaining work by owner.

## Trace workload status

The raw dataset files are local inputs and are not committed as a processed,
canonical MEC dataset. The Alibaba 2017 loader joins `batch_task.csv` and
`batch_instance.csv`, applies the documented mapping/rejections, and returns
the shared `Task` objects:

```bash
python scripts/prepare_trace_sample.py data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv
python scripts/run_trace_workload.py data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv --max-tasks 20
```

See [`docs/trace-data-notes.md`](docs/trace-data-notes.md) for the verified
schema mapping, synthetic MEC assumptions, current audit counts, and
chronological train/validation/test splitting. The RL training/compare commands
also use this loader with `--source alibaba2017`; inputs are parsed again on
each run, and no reusable processed cache is currently written.

## Run a trace workload on the multi-device topology

With the local Alibaba 2017 files, run:

```bash
python scripts/run_trace_workload.py \
  data/raw/batch_task.csv \
  --instances data/raw/batch_instance.csv \
  --source alibaba2017 \
  --max-tasks 20 \
  --topology-config configs/alibaba-multi-topology.yaml
```

The loader assigns accepted tasks round-robin to `device-1`, `device-2`, and
`device-3`. The topology then maps devices 1 and 2 to `edge-1` and device 3
to `edge-2`; these IDs select actual SimPy resources. The command writes
`task_results.csv` and `aggregate_metrics.csv` under
`results/trace-multi-topology/`. Use `--source synthetic` without an input
file to run the same topology with the preserved synthetic generator.

Student 1 owns the simulator and physical workload/resource model. Student 2's
integrated RL adapters satisfy the protocols in
`src/mec_offloading/domain/ports.py` and train/evaluate through that same
backend. The simulator/domain layers must not import Gymnasium or
Stable-Baselines3.

The exact acceptance criteria and task owners are in `docs/task-board.md` and `docs/first-week-plan.md`.

## Repository structure

```text
hierarchical-mec-offloading/
├── .github/                 # CI, issue and pull-request templates
├── configs/                 # Versioned experiment parameters
├── data/                    # README only; raw trace files remain untracked
├── docs/                    # Architecture, interfaces, decisions, rubric, protocol
├── scripts/                 # Reproducible commands and smoke tests
├── src/mec_offloading/
│   ├── domain/              # Shared tasks, states, actions and formulas
│   ├── simulation/          # Student 1: selected simulator adapter/backend
│   ├── baselines/           # Student 1: Random, Greedy, EDF
│   ├── rl/                  # Student 2: Gymnasium envs, agents and training
│   ├── coordination/        # Joint: feedback and message accounting
│   └── evaluation/          # Joint: experiment runner and aggregations
├── tests/
├── artifacts/               # Local model/figure outputs; large files ignored
└── pyproject.toml
```

Generated experiment outputs are written under `results/`, which is ignored by
Git. Do not treat development smoke-test results as final scientific findings.

## Run the Student 1 simulation smoke test

Install the declared dependencies with `pip install -e ".[dev]"`, then run:

```bash
python scripts/run_simulation_smoke.py
```

The script uses a seeded synthetic workload and prints measured local,
device-to-edge and device-to-edge-to-cloud results. It is a smoke test, not an
experimental claim; compare policies only with the reproducible experiment
runner and matching workloads.

## Demonstrate the multi-device, multi-edge topology

Run the deterministic manual demonstration:

```bash
python scripts/run_multi_topology_demo.py
```

It uses three devices, two independently queued Edge servers, and one shared
Cloud scheduler. Devices 1 and 2 share Edge 1, while Device 3 uses Edge 2.
The output is generated from actual SimPy results and includes task source
identifiers, routes, completion times, latency, device energy, deadlines, and
the Cloud execution order across both Edge origins. This is an architecture
demonstration, not a performance benchmark: the workload and policy choices
are deliberately small and deterministic.

## Run reproducible baseline experiments

Run one workload for each seed and compare every baseline on the matching
workload:

```bash
python scripts/run_baselines.py --seeds 1 2 3
```

Inspect:

```bash
column -s, -t < results/dev-smoke/aggregate_metrics.csv
column -s, -t < results/dev-smoke/aggregate_statistics.csv
```

The protocol, baseline definitions, scenario assumptions, metric denominators
and current topology limitation are documented in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Quality gates

Before merging a pull request:

```bash
ruff check .
pytest
```

A change is not complete merely because the code runs on its author's machine. Include tests or a reproducible manual check, document assumptions, and update the relevant interface/experiment document.

## Final report

Follow the professor's required seven-section article structure. See `docs/report_outline.md` and `docs/evaluation_plan.md`. Every result figure must be discussed in the text, not just inserted.
