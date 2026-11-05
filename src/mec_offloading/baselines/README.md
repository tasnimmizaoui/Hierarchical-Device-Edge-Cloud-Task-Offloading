# Student 1 — baseline policies

Baseline policies provide fixed, interpretable references for the later RL
experiments. They use the same policy protocols as RL adapters, so the
simulator does not need separate code paths for baseline and learned policies.

## Implemented policies

| Policy | Device decision | Edge decision | Cloud rule |
|---|---|---|---|
| `LocalOnlyPolicy` | Always local | Not used | Not used |
| `GreedyNearestDevicePolicy` | Compare estimated local and edge latency | Not used | Not used |
| `AlwaysEdgePolicy` | Always offload | Configured separately | Configured separately |
| `EdgeExecutePolicy` | Not used | Execute at edge | Not used |
| `EdgeForwardPolicy` | Not used | Forward to cloud | Configured separately |
| `RandomDevicePolicy` | Seeded random action | Not used | Not used |
| `RandomEdgePolicy` | Not used | Seeded random action | Not used |
| `RandomCloudPolicy` | Not used | Not used | Seeded random rule |
| `EDFCloudPolicy` | Not used | Not used | Earliest absolute deadline first |
| `FIFOCloudPolicy` | Not used | Not used | Queue arrival order |

These classes are in `policies.py`. The experiment labels that combine them
are documented in `src/mec_offloading/evaluation/README.md` and
`docs/experiment_protocol.md`.

## Important distinction

Placement and scheduling are separate:

- placement decides local, edge, or cloud routing;
- scheduling decides which already-routed cloud task executes next.

EDF is therefore a scheduling rule, not a complete placement policy. The
controlled tests verify that EDF and FIFO can produce different execution
orders when deadlines and queue contention require it.

## Reproducibility

Random policies receive independent `random.Random` instances derived from the
experiment seed. Never use the module-level random generator in a policy.
