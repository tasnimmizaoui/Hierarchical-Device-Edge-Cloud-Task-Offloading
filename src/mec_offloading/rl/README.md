# Integrated reinforcement learning

- `dqn.py`: Double-DQN, replay, target network, checkpoints, and loss logs.
- `credit.py`: multi-objective rewards and terminal per-task attribution.
- `observations.py` and `spaces.py`: normalized local observations and the
  observation/action contract for each agent.
- `policies.py` and `integrated.py`: Device, Edge, Cloud, and centralized
  adapters connected to the canonical SimPy backend.
- `scenarios.py`: temporary synthetic load and network variations.
- `envs/device_mock_env.py`: legacy Gymnasium smoke test; do not use it for
  project performance results.

The integrated transition formulation makes one routing or scheduling decision
per task. Terminal and next-decision credit modes are available; this is not a
multi-step `gymnasium.Env`.

## Current status

`envs/device_mock_env.py` is an API and training smoke test only. Its rewards,
latencies, and energy values are synthetic proxies and must not appear as
project evaluation results.

The integrated workflow implements:

1. device decisions: local execution or edge offloading;
2. edge decisions: edge execution or cloud forwarding;
3. cloud scheduling: EDF, shortest-processing-time, or FIFO rule selection;
4. hierarchical coordination and feedback;
5. a centralized RL baseline for comparison.

The training and comparison scripts can use synthetic scenarios or Alibaba
2017 tasks with an explicit multi-device topology. The centralized policy is a
comparison policy with global observations; it is not one of the separate
hierarchical agents.

## Integration rules

RL adapters must implement the framework-independent protocols in
`mec_offloading.domain.ports`:

- `DevicePolicy.choose(...)`;
- `EdgePolicy.choose(...)`;
- `CloudSchedulingPolicy.choose_rule(...)`.

Keep Gymnasium and Stable-Baselines3 imports inside this package. The domain
and simulation layers must not depend on either framework. Convert model output
to the shared `DeviceAction`, `EdgeAction`, and `CloudSchedulingRule` values.

## Smoke checks

```bash
python scripts/check_environment.py
pytest -q
pytest -q tests/test_mock_env_optional.py
python scripts/train_device_mock.py --timesteps 2000
```

The mock environment is an API smoke test only. Use
`scripts/train_integrated_rl.py` and
`scripts/compare_integrated_policies.py` for experiments on the canonical
SimPy backend. Checkpoint metadata records the workload source, topology,
observation version, reward version, seeds, and training configuration.
