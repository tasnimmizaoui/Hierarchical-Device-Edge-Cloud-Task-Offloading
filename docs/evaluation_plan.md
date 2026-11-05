# Evaluation plan mapped to the professor's rubric

## Understanding and mandatory implementation — 30%

Evidence: exact three-tier architecture; all three decision components; state/action/reward definitions; task lifecycle (generation, upload, queue, execute, return, complete/fail); consistent units; end-to-end integration; runnable scenario. A diagram or trained checkpoint alone is not implementation evidence—show that actions change placement and scheduling in the simulator.

## Comparative analysis and experimental rigor — 25%

Compare Random placement, Greedy/Nearest feasible placement, EDF scheduling, Centralized RL and Hierarchical RL. Clarify that EDF is a scheduling rule: define placement separately and schedule eligible tasks with EDF. Keep workload, capacities, network conditions, stopping rule and seeds aligned across policies.

Metrics:
- Mean and 95th-percentile end-to-end latency.
- Deadline success and miss rate.
- Device energy per generated task and per completed task (state the denominator).
- Communication delay, messages/bytes and coordination overhead.
- Throughput and resource utilization as supporting metrics.

Use repeated seeds and report mean plus variability (standard deviation or confidence interval). Include at least one load/workload sweep and the no/periodic/frequent feedback comparison. Explain why trends happen and discuss where RL performs worse.

## Code quality — 10%

- Clear modules, type hints and docstrings for non-obvious assumptions.
- Centralized versioned configuration; no scattered experimental constants.
- Unit tests for formulas, task routes, deadline boundaries, queue ordering and reproducibility.
- GitHub Actions CI for lint and tests.
- README setup and exact reproduction commands.
- Seeds, config snapshot, commit hash and package versions in each run.

## Written article — 20%

Use the course's exact article order: Title/Authors/Abstract (one 150–250-word paragraph) and 4–6 keywords; Introduction; Related Work; System Model/Proposed Approach; Implementation & Performance Evaluation; Conclusion & Future Work; IEEE-numbered References. Use `docs/report_outline.md`. Every figure must be cited and interpreted in prose. Distinguish trace-derived fields from synthetic assumptions.

## Oral defense — 15%

Both students must explain the hierarchy, state/action/reward per tier, latency/energy/deadline accounting, baseline fairness, experimental reproducibility, limitations and trade-offs. Prepare a short run showing one task's path and measured outcome.

## Extension bonus — up to +3/20

Only after mandatory tasks are complete: federated policy aggregation device→edge→cloud. Measure model quality and communication savings. Do not claim formal privacy guarantees merely because raw data is not shared. Final grade remains capped at 20/20.
