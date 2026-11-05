# Étudiante 2 — RL intégré V2

- `dqn.py` : Double-DQN, replay, target network, checkpoints, loss logs.
- `credit.py` : récompense multi-objectif, attribution *terminale* par tâche.
- `observations.py` + `spaces.py` : vecteurs locaux normalisés et contrat
  d'observation/action pour chaque agent.
- `policies.py` + `integrated.py` : agents Device/Edge/Cloud et centralisé,
  raccordés au backend SimPy de l'étudiante 1.
- `scenarios.py` : variations synthétiques temporaires de charge et réseau.
- `envs/device_mock_env.py` : ancien smoke test Gymnasium, **ne pas utiliser**
  pour valider les performances du projet.

Les transitions intégrées sont **terminales** (un choix par tâche). Un vrai
`gymnasium.Env` multi-étapes reste un travail d'interface avec l'étudiante 1.
Voir `docs/ETUDIANTE_2_V2_FR.md`.
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
