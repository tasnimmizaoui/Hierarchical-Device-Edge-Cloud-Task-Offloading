# V2 — amélioration du travail de l'étudiante 2

Le dépôt principal et le simulateur SimPy de l'étudiante 1 sont **conservés**.

## Réalisé maintenant

- Double-DQN Device / Edge / Cloud et Centralized conservés et harmonisés.
- Attribution de récompense **à la tâche qui a provoqué la décision** ;
  transitions terminales plutôt qu'état suivant d'une autre tâche.
- Coordination temporelle : valeurs de congestion rendues visibles seulement
  après un délai de propagation/sérialisation (non bloquant, sans contention).
- Registre des observations/actions, vérifications d'entrées et de transitions.
- Entraînement séparant seeds de train et validation ; meilleure sauvegarde et
  dernier checkpoint ; suivi des pertes, exploration, distribution des actions.
- Scénarios synthétiques nominal, low_load, high_load et congested_network.
- Comparaison Random, Greedy/FIFO, Greedy/EDF, Centralized RL et 3 modes de
  feedback hiérarchique sur les **mêmes tâches** pour chaque seed/scénario.
- Protection contre réutilisation de seeds de train/validation au test.
- Dashboard Streamlit multi-onglets lisant les CSV de résultats réels.
- Tests unitaires ajoutés ; version des observations changée (`simpy-v3-causal-feedback`).

## À faire avec l'étudiante 1

- Livraison et adaptation de traces Alibaba/Google avec splits temporels.
- Vrai système multi-devices / multi-edges.
- Wrapper Gymnasium pas-à-pas, crédit RL multi-étapes.
- Message-control-plane avec contention / événements SimPy dédiés.
- Expériences complètes, plusieurs répétitions, discussion statistique.

## Vérifications

- **24 tests unitaires autonomes réussis** (sans SimPy).
- Compilation Python réussie.
- Suite intégrée **non exécutée** : SimPy absent dans l'environnement de
  livraison, et accès aux serveurs de paquets non disponible.
- Aucun gain de performance n'est présenté comme démontré.

Voir `docs/ETUDIANTE_2_V2_FR.md` pour les commandes et les limites détaillées.
