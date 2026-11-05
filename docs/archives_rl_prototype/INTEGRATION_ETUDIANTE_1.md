> **ARCHIVE DU PROTOTYPE SYNTHÉTIQUE :** non contractuel pour la version SimPy intégrée.

# Contrat technique : intégration Étudiante 1 ↔ Étudiante 2

## 1. Répartition / dépendances

**Étudiante 1 fournit** : `Task`/trace loader, topologie, simulateur et ses `reset/submit_task/step/get_task_result`, les temps de transfert/attente/exécution, radio/CPU energy, EDF de référence, runner et baselines officiels.

**Étudiante 2 fournit** : `act(observation) -> action` pour Device/Edge/Cloud, modèle centralisé, reward (partagé), feedback, entraînement et tests d'ablation.

Ne pas changer le contrat des champs/units sans validation commune. Tous les temps en secondes, volumes en bytes, travail en MI, capacité en MI/s, énergie en joules. Pas de chemin direct Device→Cloud.

## 2. Schéma de Task à préserver

```json
{
  "task_id": 71,
  "arrival_time_s": 6.4,
  "input_size_bytes": 2000000,
  "required_work_mi": 1400,
  "result_size_bytes": 100000,
  "relative_deadline_s": 2.0,
  "priority": 0
}
```

Les nombres ci-dessus sont **un exemple de format**, pas une ligne réelle d'un trace. `deadline_absolute_s = arrival_time_s + relative_deadline_s`.

## 3. Interface attendue côté Étudiante 1

```python
class Student1Simulator:
    def reset(self, seed: int): ...
    def submit_task(self, task: dict): ...
    def step(self): ...
    def get_task_result(self, task_id: int): ...
```

On doit obtenir au minimum, pour chaque tâche, le temps de retour du résultat, le niveau d'exécution, l'énergie Device, les statistiques des queues/capacités et les états locaux observables AVANT que chaque agent décide. Il faut ajouter une entrée pour chacune des actions suivantes :

- `on_device_decision(task_id, action=0|1)` : exécution locale vs envoi à Edge.
- `on_edge_decision(task_id, action=0|1)` : exécution Edge vs transfert Cloud.
- `on_cloud_decision(task_id, action=0|1)` : FIFO vs EDF pour la file Cloud.

Les trois `on_*_decision` sont une **proposition d'interface de collaboration**, non une API déjà implémentée dans le simulateur Étudiante 1.

## 4. Ordre d'une tâche, strictement

1. Création/arrivée d'une Task et snapshot local Device.
2. Envoi éventuel de feedback Edge→Device, enregistré en nombre, bytes et délai.
3. `device_agent.act(device_obs)`.
4. Si action 0, simuler fin locale et passer à 8.
5. Sinon, transfert d'entrée Device→Edge et capture snapshot local Edge ; feedback Cloud→Edge si prévu.
6. `edge_agent.act(edge_obs)` ; si action 0, Edge calcule et passe à 8.
7. Sinon, transfert Edge→Cloud, `cloud_agent.act(cloud_obs)`, scheduler FIFO/EDF, exécution, retour Cloud→Edge→Device.
8. Enregistrer : `arrival`, `completion`, `transfer`, `queue`, `execution`, `energy`, deadline, feedback.
9. Calculer `R`, mettre à jour seulement les agents qui ont participé (récompense retardée).

Pour les architectures réelles de simulation, les messages de feedback peuvent être transmis à cadence discrète. Fixer explicitement la convention de latence selon qu'ils bloquent ou non le traitement d'une nouvelle tâche.

## 5. Variables à partager avant intégration

| ID | Décision à signer | Proposition actuelle |
|---|---|---|
| C-01 | Nom de la classe Task et identifiants | `task_id` unique de bout en bout |
| C-02 | Observations API exposées par niveau | `device_observation`, `edge_observation`, `cloud_observation` |
| C-03 | Description du temps de file | `queue_s` séparé de `execution_s` |
| C-04 | Sortie et retour du résultat | Inclure les 2 sens et les 2 liens si Cloud |
| C-05 | Énergie Device | CPU local + radio du Device, en J |
| C-06 | Deadlines | secondes relatives et absolues |
| C-07 | Cadence feedback | none/periodic(1 message update / 5 tâches par voisin)/frequent |
| C-08 | Dimension des messages | 16 payload + 16 en-tête = 32 bytes par message |
| C-09 | État centralisé | vue globale de référence, noter coût non simulé |
| C-10 | Source des tâches | trace/subset/mapping documenté par Étudiante 1 |

C-07 et C-08 sont **des paramètres paramétrables**, pas des vérités du réseau. Les valider ensemble.

## 6. Validation de bout en bout après raccordement

- [ ] Une tâche routée Device uniquement.
- [ ] Une tâche routée Device→Edge (aller + retour).
- [ ] Une tâche routée Device→Edge→Cloud (aller + retour).
- [ ] Action Cloud `FIFO` / `EDF` influence les dispatchs réels ; pas seulement une formule.
- [ ] Identifiant de Task inchangé de bout en bout.
- [ ] Même scénario, même seed, mêmes metrics pour Random / Greedy / Greedy+EDF / Centralized RL / Hierarchical RL.
- [ ] Actions invalides rejetées ; aucune donnée globale transmise à un agent local.
- [ ] Message count/bytes/delay relevés pour feedback none/periodic/frequent.
- [ ] Train/validation/test physiquement séparés (pas de fuites de tâches du trace).
- [ ] Tests de validité : temps ≥ 0, `latency = completion−arrival`, deadline success correctement calculé, somme des durées cohérente.
- [ ] Checkpoints réentraînés dans le simulateur final ; plots régénérés.

## 7. Limites connues du simulateur de démonstration

`rl_offloading/simulator.py` *ne constitue pas* un moteur d'événements discrets détaillé : les queues sont des snapshots échantillonnés indépendants. Le choix EDF estime l'attente pour la nouvelle tâche mais ne calcule pas l'effet sur la fin des autres tâches concurrentes. Le throughput est dérivé de timestamps indépendants, sans garantie de conservation des ressources. La charge est entièrement générée, pas rejouée d'un trace. Les comparaisons de ce substitut montrent seulement que le pipeline et les scripts fonctionnent.

**IMPORTANT :** L'intégration requiert un vrai traitement des files et de la causalité temporelle. Elle ne peut pas être certifiée sans le code Étudiante 1.
