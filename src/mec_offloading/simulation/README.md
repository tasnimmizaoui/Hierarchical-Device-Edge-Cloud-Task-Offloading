# Student 1 — simulation backend

The selected backend is SimPy. It implements the
`mec_offloading.domain.ports.SimulationBackend` protocol and keeps the model
small enough to validate in an oral defense.

Required behavior:

- independent device, edge, and cloud policy decisions;
- compute and network queueing;
- result-return traffic;
- task outcomes and deadline checks;
- latency, energy, and communication metrics.

## SimPy simulation backend

`SimPyBackend` is the Student 1 reference backend. It uses SimPy only as the
discrete-event kernel; the MEC model and policy interfaces remain owned by this
repository.

Each task follows one of three routes:

- device only: local CPU execution;
- device → edge: device-edge upload, edge CPU execution, result download;
- device → edge → cloud: the previous upload, edge-cloud upload, cloud queue
  scheduling, edge-cloud result transfer and result download.

Transmission time is `(decimal MB × 8) / Mbps + propagation + queue wait`.
Execution time is `MI / MIPS + queue wait`. Device energy includes local CPU
energy or radio transmit/receive energy; infrastructure energy is intentionally
not included in this first-order model.

The backend accepts `DevicePolicy`, `EdgePolicy`, and
`CloudSchedulingPolicy` independently, so an external Gymnasium/RL adapter can
be injected without importing an RL framework into the simulator.

Run the focused checks from the repository root:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 scripts/run_simulation_smoke.py
```
