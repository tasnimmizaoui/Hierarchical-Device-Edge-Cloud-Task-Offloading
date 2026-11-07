# Coordination hiérarchique V2

`feedback.py` supporte `none`, `periodic`, `frequent` et `event`.
Le mode `event` envoie un premier échantillon, puis transmet si le signal
normalisé de file change d'au moins 0,25 depuis le dernier envoi ou si
`frequency` observations se sont écoulées. Le délai de transmission
s'applique à tous les messages, y compris aux rafraîchissements.

Pour reproduire l'ablation événementielle sur Alibaba 2017 depuis la racine
du dépôt, utiliser les trois graines par défaut et la compression commune.

```bash
python scripts/run_feedback_ablation.py --feedback-modes event --arrival-compression 0.1360670304522077
python scripts/summarize_feedback_ablation.py --feedback-modes none periodic frequent event
```

Le coordinateur envoie des signaux de congestion **uniquement entre voisins**.
Le signal n'est accessible qu'après son délai de propagation + sérialisation.
Sa livraison est traitée lors de la prochaine consultation par l'agent
(lazy-delivery à l'instant simulé), sans modifier le moteur SimPy de
l'étudiante 1. Messages et bytes sont comptés par tâche.

**Limite** : la transmission ne monopolise pas une ressource réseau et ne
bloque pas l'exécution d'une tâche. Cette amélioration donne un ordre causal
correct pour la disponibilité du signal, mais **pas** une simulation complète
des coûts de communication sur les liens. À améliorer conjointement plus tard.
This package is reserved for the feedback exchanged between the three policy
levels. It is not yet implemented; do not report coordination experiments
until the message model and measurements exist.

## Planned responsibility

Coordination will define:

- which feedback fields move between device, edge, and cloud;
- who sends each message and when;
- message payload size in bytes;
- simulated communication delay;
- no-feedback, periodic-feedback, and frequent-feedback configurations;
- message count, bytes, and latency overhead.

The three configurations must use the same workload, topology, capacities,
policy checkpoints, and evaluation seeds. Only the feedback cadence may vary.

## Boundary

Coordination must not bypass the policy interfaces in
`mec_offloading.domain.ports`. It should provide local feedback to policies,
not expose the complete global simulator state to lower-level agents.
