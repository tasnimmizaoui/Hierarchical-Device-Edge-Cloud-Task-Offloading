# ADR-0001: Simulation engine selection

- **Status:** Accepted
- **Driver:** Student 1
- **Approvers:** Student 1 and Student 2
- **Decision date:** 2026-11-05

## Context

The model needs dynamic arrivals, compute and communication queues, hierarchical device/edge decisions, cloud scheduling, result-return traffic, device energy accounting and repeatable experiments. Python is preferred so the simulator, Gymnasium environments and Stable-Baselines3 can share one language.

## Candidate A — PureEdgeSim Python companion (rejected)

The upstream PureEdgeSim repository now points to a Python companion and describes it as the same simulator with matching configuration/output conventions. PureEdgeSim is MEC-focused and documents extensible task orchestration, network, topology, energy, and simulation components.

**Why it is attractive:** it may save significant work on the physical simulation model and provide a useful baseline for energy/network/task metrics.

**Main risk for this exact project:** the team must verify that the Python API permits separate device and edge decisions with local observations, not just a single centralized orchestrator. Also verify whether every required result-return/coordination metric is available or extensible.

Repository: https://github.com/CharafeddineMechalikh/PureEdgeSim_Python  
Upstream project: https://github.com/CharafeddineMechalikh/PureEdgeSim

### Source verification

The feasibility spike inspected PureEdgeSim Python commit
`0cd9874d17b75e6e0a13b6d16e5034dad50e3e07` (version 6.1.0), including
`src/pureedgesim/manager.py`, `orchestrator.py`, `nodes.py`, `network.py`,
`tasks.py`, and the extension-point tests/examples.

| Requirement | Evidence | Result |
|---|---|---|
| Devices, edge and cloud nodes | `topology.py`, `nodes.py` (`EdgeDevice`, `EdgeDataCenter`, `CloudDataCenter`) | Supported |
| Dynamic task generation/execution | `tasks.py`, `manager.py`, event engine | Supported |
| Network latency and bandwidth | `network.py` (`NetworkLink`, `InfrastructureGraph`) | Supported |
| Compute queues and execution time | `nodes.py` (`submit_task`, `start_execution`) | Supported |
| Device energy/battery | `nodes.py` (`EnergyModelComputingNode`) | Supported |
| Independent device decision | `manager.py` calls one `orchestrator.orchestrate` hook | Not supported |
| Independent edge decision | Same global orchestrator hook | Not supported |
| Independent cloud scheduling | Placement algorithm is global; no cloud queue-policy hook | Not supported |
| Local observations per agent | Orchestrator receives manager/scenario and can inspect all nodes | Not guaranteed |
| Non-invasive external policies | `Simulation.set_custom_orchestrator` exists, but only for one global policy | Partial |

PureEdgeSim therefore satisfies the physical simulation concerns but not the
hierarchical decision contract. Adding the missing boundaries would require
changing its manager event flow and queue implementation, which is more
invasive than implementing the small required model behind our existing
framework-independent protocols.

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

## Decision record

- **Selected engine:** SimPy 4.x, used as an event-driven kernel behind
  `mec_offloading.domain.ports.SimulationBackend`.
- **Python version:** Python 3.11 is the project target; the current validation
  interpreter is Python 3.10, which is compatible with the backend code.
- **PureEdgeSim example run:** source tests and extension-point examples were
  inspected at the pinned commit; no installation into this project was needed.
- **Device, edge and cloud policy hooks:** implemented as three separate
  protocol arguments in the selected backend.
- **Network/queue/compute/energy metrics:** implemented explicitly in the
  project backend with documented first-order assumptions.
- **Integration effort:** small, auditable backend; no upstream fork and no
  RL-framework dependency.
- **Reason for rejecting PureEdgeSim:** its supported extension point is a
  single global orchestrator, which cannot demonstrate independent
  device-level and edge-level decisions with restricted observations.
