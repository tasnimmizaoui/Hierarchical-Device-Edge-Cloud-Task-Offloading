# Hierarchical MEC Task Offloading — Project Repository

**INSAT · Futurs Réseaux Mobiles · 2026–2027**  
**Team:** Student 1 — Simulation & Evaluation · Student 2 — RL & Coordination

## Project goal

Implement and evaluate a three-level task-offloading architecture:

- **Device:** execute locally or offload to a nearby edge.
- **Edge:** execute locally or forward a task to Cloud/HPC.
- **Cloud/HPC:** schedule tasks received from edge nodes.

The mandatory evaluation compares Random, Greedy/Nearest, EDF, Centralized RL and Hierarchical RL using end-to-end latency, device energy, deadline satisfaction and coordination overhead.

## Scope rule

Complete the mandatory three-agent system, baselines and reproducible experiments before attempting federated learning. The bonus extension is not part of the initial critical path.

## Team ownership

| Student | Owns | Main outputs |
|---|---|---|
| **Student 1 — Simulation & Evaluation** | Simulator feasibility and implementation, workload adapter, topology, compute/network queues, energy and latency model, Random/Greedy/EDF baselines, experiment runner and metric validation | Working simulator, trace/synthetic workload pipeline, baseline policies, metric tests, reproducible CSV runs |
| **Student 2 — RL & Coordination** | Gymnasium environments, device/edge/cloud agents, reward design, training, feedback protocol and centralized RL baseline | Three policy agents, model checkpoints, training configs/logs, centralized RL, coordination variants |
| **Both** | Freeze interfaces, integration, end-to-end tests, experiment protocol, result analysis, IEEE article, oral presentation and demo | Integrated system and complete submission |

Ownership is primary responsibility, not exclusive knowledge. Both students must be able to explain and reproduce the complete system.

## Technology choices — provisional until the simulator spike

- **Python 3.11** for a consistent development environment.
- **PureEdgeSim Python companion:** first candidate because it is MEC-oriented and has built-in infrastructure, networking, energy, task orchestration and metrics. It is only selected if the feasibility spike verifies the policy hooks required by this project.
- **SimPy:** fallback if PureEdgeSim cannot expose the device and edge decision points cleanly. SimPy is a discrete-event simulation library, not a complete MEC simulator; the team must implement and validate its own network, resource and energy models.
- **Gymnasium** for the RL environment contract.
- **Stable-Baselines3** for DQN/PPO experiments.
- **NumPy, Pandas, Matplotlib, PyYAML** for computation, data and analysis.
- **pytest + Ruff + GitHub Actions** for tests, linting and reproducibility checks.

Read `docs/decisions/0001-simulator-selection.md` before committing to either engine. Do not build two complete simulators.

## Create the GitHub repository

1. On GitHub, create a repository named `hierarchical-mec-offloading` (Private is a sensible default for an active class project; make it public later if your course permits).
2. Create it **empty**: do not pre-add a README, `.gitignore`, or license because this project folder already contains them.
3. Extract this starter archive. Open a terminal in the extracted `hierarchical-mec-offloading` folder.
4. Initialize and push the local project:

```bash
git init -b main
git add .
git commit -m "chore: initialize hierarchical MEC project"
git remote add origin https://github.com/<GITHUB-OWNER>/hierarchical-mec-offloading.git
git push -u origin main
```

Replace `<GITHUB-OWNER>` with the account or organization that owns the repository.

5. Invite Student 2 as a repository collaborator: **Settings → Collaborators**. Both students should use their own GitHub accounts.
6. Keep `main` stable. Work in short-lived branches and merge through pull requests. Enable GitHub Actions and review the CI result before merging.
7. Create GitHub Issues from `docs/task-board.md`. Apply labels such as `student-1`, `student-2`, `shared`, `simulation`, `rl`, `baseline`, `experiment`, `report`, `P0`, and `P1`.

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

## First parallel tasks

Both students can start immediately after agreeing to `docs/interfaces.md`:

- **Student 1:** complete the PureEdgeSim Python feasibility spike; inspect the Alibaba trace schema; implement and test the metric formulas; create a minimal simulator spike.
- **Student 2:** run the mock Gymnasium environment; validate it with Gymnasium/SB3; train a tiny DQN; keep the environment marked as a mock until Student 1's simulation adapter is ready.
- **Together:** freeze units, task fields, action semantics and the policy interfaces. Do not change these independently.

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

## Run the Student 1 simulation smoke test

Install the declared dependencies with `pip install -e ".[dev]"`, then run:

```bash
python scripts/run_simulation_smoke.py
```

The script uses a seeded synthetic workload and prints measured local,
device-to-edge and device-to-edge-to-cloud results. It is a smoke test, not an
experimental claim; compare policies only with the reproducible experiment
runner and matching workloads.

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
