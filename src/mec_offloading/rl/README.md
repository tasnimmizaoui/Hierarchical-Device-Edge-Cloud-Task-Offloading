# Student 2 — RL layer

This package belongs to Student 2. It contains Gymnasium environments, RL
adapters, training utilities, and the centralized RL comparison.

## Current status

`envs/device_mock_env.py` is an API and training smoke test only. Its rewards,
latencies, and energy values are synthetic proxies and must not appear as
project evaluation results.

The integrated RL work should implement:

1. device decisions: local execution or edge offloading;
2. edge decisions: edge execution or cloud forwarding;
3. cloud scheduling: EDF, shortest-processing-time, or FIFO rule selection;
4. hierarchical coordination and feedback;
5. a centralized RL baseline for comparison.

## Integration rules

RL adapters must implement the framework-independent protocols in
`mec_offloading.domain.ports`:

- `DevicePolicy.choose(...)`;
- `EdgePolicy.choose(...)`;
- `CloudSchedulingPolicy.choose_rule(...)`.

Keep Gymnasium and Stable-Baselines3 imports inside this package. The domain
and simulation layers must not depend on either framework. Convert model output
to the shared `DeviceAction`, `EdgeAction`, and `CloudSchedulingRule` values.

## Suggested workflow

```bash
python scripts/check_environment.py
pytest -q
pytest -q tests/test_mock_env_optional.py
python scripts/train_device_mock.py --timesteps 2000
```

Start with one environment and one adapter, add tests, then integrate the
frozen policy into `SimPyBackend`. Do not train all three agents before each
individual policy boundary is tested.
