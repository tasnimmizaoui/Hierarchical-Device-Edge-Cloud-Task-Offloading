# Intégration RL.zip → Hierarchical-Device-Edge-Cloud-Task-Offloading

## Choix retenu

**Le dépôt Hierarchical-Device-Edge-Cloud-Task-Offloading reste la base.** Son
`src/mec_offloading/simulation/simpy_backend.py` est la **seule simulation
utilisée pour l'entraînement intégré et les comparaisons**. Le simulateur
`rl_offloading/simulator.py` de RL.zip était un *surrogate* à événements non
corrélés : il ne doit pas servir aux mesures finales. La reprise porte sur
l'algorithme Double-DQN et les concepts d'observations, de récompense et de
feedback, pas sur ses résultats antérieurs.

## Provenance et correspondance des fichiers

| RL.zip | Après fusion | Changement |
|---|---|---|
| `rl_offloading/dqn.py` | `src/mec_offloading/rl/dqn.py` | Réemploi Double-DQN, replay, target network, checkpoint |
| `rl_offloading/simulator.py` | **Non repris** | Remplacé par le SimPyBackend principal |
| `rl_offloading/gym_envs.py` | **Non repris** | Environnements liés au surrogate ; une intégration Gymnasium complète reste à faire |
| observations de `simulator.py` | `src/mec_offloading/rl/observations.py` | Converties vers Task/DeviceState/EdgeState/CloudState, valeurs et actions revues |
| `rl_offloading/train.py` | `scripts/train_integrated_rl.py` | Entraînement sur épisodes SimPy, résultats retardés par tâche |
| `rl_offloading/runner.py` | `scripts/compare_integrated_policies.py` | Comparaison via `mec_offloading.evaluation.runner` et SimPy |
| coordination / messages | `src/mec_offloading/coordination/feedback.py` | Feedback adjacent limité ; comptes de messages, bytes, délai théorique |
| `web_app.py` | `web_app.py` (nouveau) | Visualiseur CSV SimPy, aucune simulation substitutive |

## Contrat canonique (conserver celui du dépôt de base)

- `Task.task_id: str`, `device_id: str`.
- `input_size_mb`, `result_size_mb`: **MB décimaux**, et non bytes.
- `required_mi`: millions d'instructions ; capacité : MIPS.
- `deadline_s`: **relative** ; échéance absolue : `arrival_time_s + deadline_s`.
- Réseau : `bandwidth_mbps` (**megabits/s**) ; pas bytes/s.
- Device actions : 0 = local, 1 = Edge.
- Edge actions : 0 = Edge, 1 = Cloud/HPC.
- Cloud actions : **0 = EDF, 1 = SPT, 2 = FIFO** (contrat du dépôt de base).
- RL centralisé : **0 = Device, 1 = Edge, 2 = Cloud** ; Cloud EDF fixe. Cela
  garantit un comparateur centralisé de placement sans essayer de changer une
  discipline de queue individuelle en conflit avec d'autres tâches en file.

**Attention :** les observations `simpy-v2` et les actions du Cloud et de
l'agent centralisé **ne sont pas compatibles avec les anciens checkpoints** de
RL.zip. Réentraîner les quatre agents sur le moteur SimPy est obligatoire.
Aucun checkpoint du surrogate n'a été inclus dans ce dépôt.

## Commandes

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,rl,web]'
pytest -q
python scripts/run_baselines.py --seeds 1 2 3
python scripts/train_integrated_rl.py --episodes 12 --tasks 100
python scripts/compare_integrated_policies.py --seeds 10001 10002 10003 --tasks 100
streamlit run web_app.py
```

Sous PowerShell, remplacer les trois premières lignes par :

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,rl,web]"
```

CSV créés sous `results/integrated-comparison` : `task_results.csv`,
`aggregate_metrics.csv`, `aggregate_statistics.csv`, `metadata.json`.

## Validation et limites actuelles (important pour l'article)

- Le workload utilisé par les scripts intégrés est **synthétique** via
  `generate_synthetic_tasks`, **pas** un sous-ensemble Alibaba/Google.
- Le moteur SimPy gère causalité, ressources CPU, transferts et files partagées,
  mais le topo de référence est **1 device / 1 edge / 1 cloud**. Il faut étendre
  aux multiples devices/edges et documenter les hypothèses réseau.
- Les signaux Cloud→Edge et Edge→Device sont propagés par des messages
  **modélisés non bloquants**. Le coût en messages/bytes et le délai théorique
  sont enregistrés, mais le délai de feedback n'allonge pas la latence finale.
  Pour une expérience sur le compromis communication-latence, il faudra faire
  circuler réellement les messages dans le moteur d'événements SimPy.
- Les observations des agents locaux ne reçoivent pas d'état global brut : les
  congestions distantes ne passent que par le coordinateur. L'agent centralisé
  utilise un hook explicite `choose_global` dans SimPyBackend.
- L'entraînement utilise une **approximation DQN par visites successives** :
  chaque transition reçoit une récompense après fin de tâche et son état suivant
  est la prochaine observation de ce niveau. Ce n'est pas une preuve de
  convergence ni une formulation Gymnasium complète du système multi-agent.
- Une baseline Greedy utilise des **estimations** de délai pour décider, mais
  toutes les politiques sont finalement mesurées par **le même** backend SimPy.
- La composante Cloud travaille sur la queue réelle et choisit EDF/SPT/FIFO ;
  elle n'optimise pas arbitrairement un ID de tâche.
- Les états de queue fournissent aujourd'hui une estimation rudimentaire de
  `queue_wait_s`. Les deadlines, les pics de charge, les valeurs énergétiques
  et les régimes réseau doivent être vérifiés pour le protocole final.
- Séparer les tâches de train/validation/test via fichiers de trace distincts,
  ajouter les sweeps faible/moyenne/forte charge et réseau bon/modéré/congestionné,
  vérifier les seeds et répéter les mesures. Les scripts de développement
  n'implémentent pas encore tout ce protocole.
- Le mécanisme fédéré est **hors périmètre** de cette version.

Ne pas présenter les métriques du prototype RL.zip ou de ces tests synthétiques
comme des gains démontrés sur un trace public ou un système MEC réel.

## Mise à jour V2 (05/11/2026)

Ce document décrit la fusion initiale. La V2 améliore les aspects suivants :

- `rl/credit.py` remplace l'approximation « transition vers la tâche suivante »
  par des transitions **terminales par tâche**. Le paragraphe ancien sur les
  visites successives ci-dessus est désormais **historique**, pas le code V2.
- `coordination/feedback.py` ne rend plus les messages immédiatement visibles :
  la valeur est réceptionnée après délai à la prochaine décision (sans
  contention du canal de contrôle).
- `scripts/train_integrated_rl.py` sépare train/validation, suit les pertes,
  epsilon/actions, et sauvegarde les meilleurs checkpoints.
- `scripts/compare_integrated_policies.py` réalise une ablation sur les modes
  de feedback avec des seeds de test disjointes.
- `rl/observations.py` devient `simpy-v3-causal-feedback`. **Réentraîner les
  modèles après la mise à jour**, les anciens checkpoints ne sont pas acceptés.

Consulter **[`ETUDIANTE_2_V2_FR.md`](ETUDIANTE_2_V2_FR.md)** pour les instructions
actuelles, les critères de complétion et les limites scientifiques.
