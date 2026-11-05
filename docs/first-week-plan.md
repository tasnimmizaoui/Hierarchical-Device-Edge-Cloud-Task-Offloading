# First-week execution plan — two students

Use sessions rather than fixed dates. Enter actual dates in GitHub Issues after confirming the course deadline and other coursework.

## Session 0 — one joint setup session

- [ ] Create the GitHub repository and invite both students.
- [ ] Both students clone the same repository and install Python 3.11 environments.
- [ ] Read `docs/architecture.md` and `docs/interfaces.md` together.
- [ ] Agree the units, `Task` fields, device/edge/cloud action meanings, observation fields and reward components.
- [ ] Create the initial Issues from `docs/task-board.md`.
- [ ] Agree that `main` is protected by review; use branches and pull requests.

**Exit condition:** both can run `pytest`; there are no unresolved disagreements about task units or the meaning of actions.

## Student 1 — Simulation & Evaluation

### S1-01: simulator feasibility spike (P0)

Time-box this to one focused work session. Start with the PureEdgeSim Python companion linked in the simulator decision record.

1. Clone it separately; do not copy its source into this repository.
2. Run one unmodified example with its documented setup.
3. Find the task generation, task placement/orchestration and scheduling extension points.
4. Verify whether device and edge decisions can be made separately with only their permitted local observations.
5. Check whether transfer delay, queue wait, execution time, return-path traffic, battery/device energy and task failure information can be observed or extended.
6. Export one run and inspect its output columns.
7. Record version/commit, commands, results, limitations and the decision in `docs/decisions/0001-simulator-selection.md`.

**Pass condition:** a small custom policy can be called at the required hierarchy levels without invasive changes, and required metrics are accessible or cleanly extensible. If not, select SimPy and explicitly implement the missing models.

### S1-02: workload schema and mapping (P0)

- Inspect the official Alibaba Cluster Trace v2018 schema before downloading large files.
- Identify which table(s) contain task arrival and resource-demand fields needed by your model.
- Select a manageable subset and write a streaming/chunked preprocessing script.
- Preserve raw values and document every transformation.
- Mark generated deadlines, device assignment, input/output size, battery and radio conditions as synthetic unless the trace directly supplies them.
- Produce a small anonymized/derived sample for tests if licensing and dataset terms permit; do not commit the full raw trace.

**Pass condition:** a documented CSV/Parquet schema maps into the shared `Task` model with units and provenance.

### S1-03: model formula tests and minimal topology (P0)

- Implement/test transmission and compute time formulas.
- Create one device, one edge and one cloud/HPC test topology.
- Demonstrate one deterministic task through local, edge and cloud routes with fixed policies.
- Ensure result-return transfer is not silently omitted.

**Pass condition:** unit tests pass and each route returns an explicit `TaskResult`.

## Student 2 — RL & Coordination

### S2-01: mock device Gymnasium environment (P0)

- Install `.[dev,rl]`.
- Run `python scripts/check_environment.py` and `pytest`.
- Run Gymnasium's `check_env` against `DeviceOffloadingMockEnv`.
- Read the observation ordering and discrete action meanings in `docs/interfaces.md`.
- Train a tiny DQN using `python scripts/train_device_mock.py --timesteps 2000`.
- Save mock training artefacts locally; do not commit model binaries.

**Pass condition:** the environment passes API checks, DQN training runs, and the code clearly labels all rewards/latency/energy in this mock as synthetic proxies.

### S2-02: design the three agent adapters (P1)

- Define device and edge policy classes that implement the shared `choose(...)` methods.
- Define the cloud scheduler adapter implementing `choose_rule(...)`.
- Keep Stable-Baselines3 imports inside the RL module; simulation/domain modules must not depend on SB3.
- Draft the three reward components and normalization constants; propose values but do not tune against test data and call them results.

**Pass condition:** the RL layer can be instantiated and tested against mock observations without importing the selected simulator.

## Joint integration checkpoint

At the end of the first week:

- [ ] Decide PureEdgeSim Python vs SimPy using the written evidence.
- [ ] Freeze Interface v1 (changes thereafter require a PR reviewed by both).
- [ ] Demonstrate a mock device action plus one end-to-end fixed-policy task route.
- [ ] Agree the first experimental scenario and seed strategy.
- [ ] Reassign work if one student's blocked time or task load is materially higher.

## Do not do yet

- Do not download the complete 98 GB Alibaba v2018 archive.
- Do not implement federated RL.
- Do not train all three agents simultaneously before each local environment is tested.
- Do not build a dashboard before the simulator produces reliable metrics.
- Do not tune final reward weights on the same seeds used for final evaluation.
