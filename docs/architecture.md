# System architecture — initial proposal

```mermaid
flowchart TD
    T[Task arrives at device] --> D[Device policy / RL agent]
    D -->|Local| DL[Device CPU queue]
    D -->|Offload| E[Selected edge server]
    E --> EA[Edge policy / RL agent]
    EA -->|Execute| EQ[Edge CPU queue]
    EA -->|Forward| C[Cloud / HPC queue]
    C --> CA[Cloud scheduler / RL agent]
    CA --> CQ[HPC execution]
    DL --> RET[Return result to device]
    EQ --> RET
    CQ --> RET
    RET --> M[Latency, energy, deadline, traffic metrics]
    M -. local reward / feedback .-> D
    M -. local reward / feedback .-> EA
    M -. cloud reward .-> CA
```

## Decision boundaries

- Device chooses only local execution or a nearby edge. It does not directly choose cloud.
- Edge chooses only edge execution or cloud/HPC forwarding.
- Cloud schedules the tasks that arrive from edge nodes.
- Each agent sees only its permitted local observation plus deliberately defined feedback fields. Do not pass global state to lower agents in the hierarchical variant.
- Model device-edge and edge-cloud links separately, including uplink and result-return/downlink traffic if available.
- A deadline is a duration from arrival, not an absolute timestamp in the input.
- Begin with independent tasks. DAG tasks are optional unless required by the professor.

## Cloud action-space proposal

For a fixed discrete DQN action space, the cloud agent chooses a queue discipline at a dispatch point:

- `0`: EDF
- `1`: Shortest Processing Time
- `2`: FIFO

This is a learned selection among scheduling rules. It is not arbitrary selection by task ID. Report that distinction accurately.

## Training strategy

1. Train each agent in a controlled environment while the other tiers use fixed policies.
2. Evaluate each frozen agent in the complete hierarchy.
3. Optionally fine-tune/alternate training to investigate interaction and non-stationarity.
4. Compare all methods on matching scenarios, workloads, resource capacities and evaluation seeds.

This is a practical training schedule, not a proof of global optimality.
