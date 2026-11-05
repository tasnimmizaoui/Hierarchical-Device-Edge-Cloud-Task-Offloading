# Étudiante 2 — V2 du module RL et coordination (05/11/2026)

## Portée

**Référentiel conservé :** `Hierarchical-Device-Edge-Cloud-Task-Offloading`.
L'étudiante 1 reste responsable de la simulation SimPy, des traces publiques,
des topologies, des métriques physiques et des baselines. Cette version modifie
l'**adaptation RL, la coordination et les scripts d'entraînement/évaluation**, et
ajoute des tests et une interface de consultation CSV. **Pas de simulateur
alternatif.**

## Nouveautés réellement codées

1. **Attribution de crédit correcte par tâche :** la décision de l'agent est
   associée au résultat final de **la même tâche**. `done=True` et état suivant
   nul pour chaque décision. Aucune transition n'enchaîne des observations de
   tâches différentes. Les crédits ne sont distribués **qu'après la fin de
   l'épisode** et uniquement aux tiers qui ont décidé pour cette tâche.
2. **Limite scientifique explicite :** cette formulation est actuellement un
   **problème contextuel à un seul pas appris avec Double-DQN**, et non un
   MDP multi-étapes Gymnasium complet. Pour le RL multi-étapes, il faudra
   qu'étudiante 1 expose un moteur qui peut être suspendu et repris à chaque
   point de décision sans perdre les files et les événements.
3. **Feedback temporel :** un message émis à `t` arrive à `t + propagation +
   serialization`. Sa valeur ne devient disponible qu'au prochain point de
   décision **postérieur à cette arrivée**. Fréquences none/periodic/frequent,
   compteurs par tâche de bytes/messages/délai cumulés, sans état global brut
   donné à l'agent Device/Edge. Messages différés **non bloquants**, sans
   simulation de contention avec les transferts de données.
4. **Registry d'espaces :** `rl/spaces.py` décrit les observations (10/10/9/15)
   et les actions (2/2/3/3) ; adaptation facultative aux objets Gymnasium.
   Ce n'est **pas** un wrapper `gymnasium.Env` multi-agent fonctionnel.
5. **Entraînement reproductible :** 4 Double-DQN, logs par épisode et tier
   (reward, loss, epsilon, actions, success, latence), seeds d'entraînement
   séparées des seeds de validation, checkpoint du meilleur score de validation
   pour le groupe hiérarchique et pour l'agent centralisé ; dernier état des
   modèles sous `last/`.
6. **Scénarios synthétiques** nommés : nominal, low_load, high_load et
   congested_network. On fait varier l'espacement des arrivées, la bande
   passante et la propagation, en utilisant le **même** moteur SimPy.
7. **Comparaison reproductible :** Random, Greedy/FIFO, Greedy/EDF,
   Centralized-RL, et Hierarchical-RL sans/avec feedback périodique/fréquent,
   avec des seeds **test** rejetées si déjà utilisées pour train/validation.
   Résultats par scénario séparés ; moyennes et écarts types sur les seeds,
   sans mélange de scénarios.
8. **Streamlit amélioré :** onglets Comparaison, Entraînement, Coordination et
   Parcours des tâches. Le dashboard lit uniquement des CSV produits par les
   scripts (aucun chiffre illustratif).

## Exécution (après installation locale de SimPy)

À la racine du dépôt, sous Python 3.11 :

```bash
python -m pip install -e '.[dev,rl,web]'
pytest -q
python scripts/train_integrated_rl.py --episodes 18 --tasks 100 --seed 42
python scripts/compare_integrated_policies.py --seeds 10001 10002 10003 --tasks 100
streamlit run web_app.py
```

La comparaison utilise les poids **sélectionnés sur validation** de
`artifacts/models/integrated/{device,edge,cloud,centralized}.pt` ; les derniers
poids sont dans `artifacts/models/integrated/last/`. Voir
`training_log.csv`, `metadata.json`, `results/integrated-comparison/*.csv`.
Pour une vérification rapide, réduire `--episodes`, `--tasks`, `--validation-tasks`
et/ou `--scenarios nominal`. L'entraînement nécessite PyTorch.

Pour une ablation sur un seul mode de feedback :

```bash
python scripts/compare_integrated_policies.py --scenarios nominal --feedback periodic
```

Attention : le feedback est **évalué à l'inférence** sans réentraîner pour
chaque mode. Les résultats mesurent donc l'effet des modes sur un même modèle,
pas le meilleur modèle spécialisé pour chaque mode.

## Ce qui manque avant le rendu du Projet 05

| Référence | Statut après V2 | Travail conjoint restant |
|---|---|---|
| S2-01 | **Partiel** | Vrai `gymnasium.Env`/API pas-à-pas sur moteur SimPy |
| S2-02 | **Implémenté, validation finale manquante** | Test convergence et robustesse sur trace réelle |
| S2-03 | **Implémenté, validation finale manquante** | Tests multi-edge et routage sous congestion |
| S2-04 | **Implémenté, validation finale manquante** | Étudier capacité/quantité de workers Cloud et choix EDF/SPT/FIFO |
| S2-05 | **Implémenté au niveau tâche** | Justifier scientifiquement poids/références et autre credit assignment |
| S2-06 | **Partiel avancé** | Mettre messages sur liens SimPy avec concurrence/état différé exact |
| S2-07 | **Partiel avancé** | Expériences longues, stabilité, comparaisons stats sur traces publiques |
| S2-08 | **Implémenté en développement** | Comparaison finale et ablations sur workloads réels |

La politique Cloud choisit aujourd'hui une discipline EDF/SPT/FIFO quand un
worker unique devient libre. L'efficacité d'un vrai ordonnanceur HPC sur
plusieurs ressources reste à modéliser et à valider avec l'étudiante 1.

## Contrat de reprise pour l'étudiante 1

Les modules conservés sans modification de logique sont :
`domain/`, `simulation/simpy_backend.py`, `baselines/`, `workloads.py`,
`evaluation/`. Pour la future topologie multi-device et le trace réel :

- Conserver les identifiants de tâche uniques, les unités et les interfaces
  `DevicePolicy.choose`, `EdgePolicy.choose`, `CloudPolicy.choose_rule`.
- Conserver `set_time(now_s)` juste avant les appels des agents Device et Edge.
  Le module feedback en a besoin pour délivrer les messages au bon instant.
- Conserver `choose_global(task,device,edge,cloud)` pour le comparateur
  centralisé, et `record_dispatch(task_id)` pour le Cloud RL.
- Maintenir la même simulation et les mêmes scénarios pour les comparaisons.
  Faire passer les détails de topologie via des états *locaux* côté Device/Edge.
- Ajouter les événements SimPy de contrôle / congestion réseau si le surcoût de
  coordination doit retarder le flux de données ou consommer la bande passante.
- Remplacer `rl/scenarios.py` synthétique par un chargeur de trace réel et des
  splits train/validation/test sur fenêtres ou travaux distincts (pas seulement
  des seeds sur la même distribution).

## État des tests à la livraison

Les tests unitaires de récompense, crédit terminal, replay DQN, observations,
coordination et mapping des actions ont été exécutés. **Les tests end-to-end
SimPy, les scripts d'entraînement/comparaison et l'interface Streamlit n'ont
pas pu être exécutés dans l'environnement de préparation**, car `simpy`,
`gymnasium` et `streamlit` n'y étaient pas installés et le réseau de paquets
était inaccessible. Vérifier localement avec les commandes ci-dessus.
Ne pas annoncer de gains chiffrés réels ou de convergence avant cette validation.
