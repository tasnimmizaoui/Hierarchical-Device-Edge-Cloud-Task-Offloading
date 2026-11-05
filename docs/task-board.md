# GitHub task board

Create one Issue per task, set a clear owner, and add an acceptance checklist. Add actual due dates only after confirming the course deadline and both students' other coursework.

| ID | Owner | Priority | Task | Acceptance criteria |
|---|---|---:|---|---|
| T-00 | Shared | P0 | Create GitHub repo, invite collaborator, activate CI | Both can clone/push branches; CI runs on PRs |
| T-01 | Shared | P0 | Freeze shared contracts | Both approve `docs/interfaces.md`; no unresolved units/action meanings |
| T-02 | Student 1 | P0 | PureEdgeSim Python feasibility spike | **Complete:** source inspected; limitation and SimPy decision recorded |
| T-03 | Student 2 | P0 | Device mock Gymnasium environment | API checker passes; DQN smoke run succeeds; marked mock |
| T-04 | Student 1 | P0 | Alibaba trace schema and mapping | Trace provenance table, selected subset, reproducible adapter spec; synthetic fields labelled |
| T-05 | Student 1 | P0 | Metric formulas and tests | **Complete:** compute/transmission/deadline formulas tested with units and boundary cases |
| T-06 | Student 1 | P0 | Minimal simulator topology | **Complete:** one device/edge/cloud; local, edge, and cloud routes tested |
| T-07 | Student 2 | P1 | Device RL adapter | DQN policy implements shared `DevicePolicy` protocol |
| T-08 | Student 2 | P1 | Edge mock environment and agent | Edge policy implements shared `EdgePolicy` protocol; seeded tests pass |
| T-09 | Student 2 | P1 | Cloud scheduling environment and agent | Cloud policy selects a scheduling rule with fixed observation/action shapes |
| T-10 | Student 1 | P1 | Random and nearest/greedy baselines | **Complete:** same simulator API; random is seeded |
| T-11 | Student 1 | P1 | EDF scheduler baseline | **Complete:** controlled test confirms earliest absolute deadline dispatch |
| T-12 | Shared | P1 | Full hierarchical integration | Task route and per-tier decisions traceable from arrival to completion |
| T-13 | Student 2 | P1 | Centralized RL baseline | Global state/action space documented and compared on same scenarios |
| T-14 | Shared | P1 | Experimental protocol v1 | **Complete for current one-device topology:** workload, scenarios, seeds, metrics and policy definitions documented |
| T-15 | Shared | P1 | Coordination overhead experiment | No/periodic/frequent feedback compared; messages/bytes/delay measured |
| T-16 | Shared | P1 | IEEE article and oral defense | Required seven sections, IEEE refs, figure interpretations and rehearsed demo |
| T-17 | Shared | Bonus | Federated extension | Open only after every mandatory P0/P1 task is complete |

Suggested labels: `student-1`, `student-2`, `shared`, `simulation`, `dataset`, `baseline`, `rl`, `coordination`, `experiment`, `report`, `P0`, `P1`, `bonus`, `blocked`.
