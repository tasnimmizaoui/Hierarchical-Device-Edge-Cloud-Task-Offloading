# ADR-0001: Simulation engine selection

- **Status:** Proposed — must be replaced by an accepted decision after the feasibility spike.
- **Driver:** Student 1
- **Approvers:** Student 1 and Student 2
- **Decision date:** TBD

## Context

The model needs dynamic arrivals, compute and communication queues, hierarchical device/edge decisions, cloud scheduling, result-return traffic, device energy accounting and repeatable experiments. Python is preferred so the simulator, Gymnasium environments and Stable-Baselines3 can share one language.

## Candidate A — PureEdgeSim Python companion (first to evaluate)

The upstream PureEdgeSim repository now points to a Python companion and describes it as the same simulator with matching configuration/output conventions. PureEdgeSim is MEC-focused and documents extensible task orchestration, network, topology, energy, and simulation components.

**Why it is attractive:** it may save significant work on the physical simulation model and provide a useful baseline for energy/network/task metrics.

**Main risk for this exact project:** the team must verify that the Python API permits separate device and edge decisions with local observations, not just a single centralized orchestrator. Also verify whether every required result-return/coordination metric is available or extensible.

Repository: https://github.com/CharafeddineMechalikh/PureEdgeSim_Python  
Upstream project: https://github.com/CharafeddineMechalikh/PureEdgeSim

## Candidate B — SimPy discrete-event model

SimPy provides a process-based discrete-event simulation engine and capacity-limited resources. It is not a complete MEC simulator. The team would implement the topology, transmission time, contention, queues, compute capacities, radio/CPU energy and failure/deadline behavior explicitly.

**Why it is attractive:** direct control of hierarchical decision points and clean Python integration.

**Main risk:** more model implementation and validation work falls on Student 1; assumptions must be carefully documented in the article.

Documentation: https://simpy.readthedocs.io/en/stable/

## Time-boxed feasibility checklist

1. Clone the PureEdgeSim Python companion separately from this project.
2. Install it in a clean Python 3.11 environment using its own instructions.
3. Run one unmodified example.
4. Locate the task generator, device decision, edge decision, cloud queue and scheduler hooks.
5. Prototype a fixed local-vs-edge device policy and edge-vs-cloud policy.
6. Verify observability of link latency/bandwidth, queue wait, compute time, device energy, return traffic and task completion/deadline metrics.
7. Inspect output schema and run reproducibility/seed controls.

## Decision rule

Choose PureEdgeSim Python only if the hooks for both lower levels are genuinely separate or can be added via supported extension points without forking or invasive patching. If the only practical hook is a global orchestrator that sees all states, use SimPy for a faithful hierarchical model rather than mislabeling centralized decisions as hierarchical.

Do not build two full simulators. If PureEdgeSim fails the spike, keep the shared domain models/ports and implement one SimPy backend behind `SimulationBackend`.

## Decision record (complete after the spike)

- Selected engine and exact version/commit:
- Python version:
- Example run command and result:
- Device policy hook found and verified:
- Edge policy hook found and verified:
- Cloud scheduler hook found and verified:
- Network/queue/compute/energy metrics available:
- Missing features and adaptations:
- Decision and rationale:
- Approved by both students on:
