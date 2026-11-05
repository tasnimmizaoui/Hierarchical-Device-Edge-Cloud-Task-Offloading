# Shared — hierarchical coordination

This package is reserved for the feedback exchanged between the three policy
levels. It is not yet implemented; do not report coordination experiments
until the message model and measurements exist.

## Planned responsibility

Coordination will define:

- which feedback fields move between device, edge, and cloud;
- who sends each message and when;
- message payload size in bytes;
- simulated communication delay;
- no-feedback, periodic-feedback, and frequent-feedback configurations;
- message count, bytes, and latency overhead.

The three configurations must use the same workload, topology, capacities,
policy checkpoints, and evaluation seeds. Only the feedback cadence may vary.

## Boundary

Coordination must not bypass the policy interfaces in
`mec_offloading.domain.ports`. It should provide local feedback to policies,
not expose the complete global simulator state to lower-level agents.
