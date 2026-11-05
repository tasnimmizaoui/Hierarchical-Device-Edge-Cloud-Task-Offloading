# Coordination hiérarchique V2

`feedback.py` supporte `none`, `periodic` et `frequent`.

Le coordinateur envoie des signaux de congestion **uniquement entre voisins**.
Le signal n'est accessible qu'après son délai de propagation + sérialisation.
Sa livraison est traitée lors de la prochaine consultation par l'agent
(lazy-delivery à l'instant simulé), sans modifier le moteur SimPy de
l'étudiante 1. Messages et bytes sont comptés par tâche.

**Limite** : la transmission ne monopolise pas une ressource réseau et ne
bloque pas l'exécution d'une tâche. Cette amélioration donne un ordre causal
correct pour la disponibilité du signal, mais **pas** une simulation complète
des coûts de communication sur les liens. À améliorer conjointement plus tard.
