# Projet 05 — synthèse de la fusion

**Dépôt principal** : `Hierarchical-Device-Edge-Cloud-Task-Offloading`.

Ce qui a été fusionné :
- DQN (device, edge, cloud) et DQN centralisé récupérés de `RL.zip`.
- Adaptateurs de données et d'actions vers le contrat existant (`src/mec_offloading/domain`).
- Exécution commune via le moteur SimPy du dépôt principal.
- Feedback adjacent (Cloud→Edge, Edge→Device), comptage messages/bytes.
- Comparateur de 5 politiques avec export CSV et interface Streamlit de consultation.
- Tests unitaires et notes d'architecture / limites / provenance du prototype.

**Pas repris comme source officielle** : le simulateur de démonstration `RL.zip`,
les métriques synthétiques préexistantes, les anciens checkpoints incompatibles.

## Commandes après installation Python 3.11 + dépendances

```bash
pip install -e ".[dev,rl,web]"
pytest -q
python scripts/train_integrated_rl.py --episodes 12 --tasks 100
python scripts/compare_integrated_policies.py --seeds 10001 10002 10003
streamlit run web_app.py
```

## Ce qu'il faut encore faire pour terminer le sujet

1. Intégrer un sous-ensemble du trace Alibaba ou Google, documenter les mappings
   entre champs fournis et champs dérivés.
2. Étendre la topologie actuelle 1 device / 1 edge / 1 cloud vers plusieurs
   devices et edges, et vérifier la répartition de charge.
3. Faire passer les messages de coordination par de vrais événements réseau
   pour mesurer le compromis qualité/latence de communication.
4. Ajouter wrapper(s) Gymnasium final(aux), tests de scheduling/concurrence
   et validation de l'estimation de queue / batterie.
5. Évaluer les 5 politiques sur les mêmes scénarios avec trains/validation/tests
   disjoints, variantes de charge et de réseau, graines répétées.
6. Refaire les courbes et rédiger l'analyse avec des résultats réellement mesurés.

Les tests unitaires des adaptateurs passent (4 tests), mais **l'exécution du
backend SimPy n'a pas pu être vérifiée dans l'environnement de création du ZIP**
car la dépendance n'était pas disponible ; elle doit être testée localement.
