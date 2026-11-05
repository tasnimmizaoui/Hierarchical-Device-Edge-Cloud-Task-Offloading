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
