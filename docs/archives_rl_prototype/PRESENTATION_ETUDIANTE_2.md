> **ARCHIVE DU PROTOTYPE SYNTHÉTIQUE :** non contractuel pour la version SimPy intégrée.

# Présentation orale — Partie Étudiante 2 (8 diapositives)

## Slide 1 — Ma responsabilité
**Titre :** Apprentissage par renforcement et coordination hiérarchique
**À l'oral :** « Ma contribution concerne les décisions intelligentes prises aux trois niveaux, la fonction de récompense, les échanges de feedback et l'évaluation contre une politique RL centralisée. Le simulateur de ressources est construit par l'Étudiante 1. »

## Slide 2 — Architecture et décisions
**Visuel :** Device (local / Edge) → Edge (Edge / Cloud) → Cloud (FIFO / EDF), retour Cloud→Edge→Device.
**À l'oral :** « Chaque agent n'observe que les informations disponibles localement, complétées par des messages explicitement échangés. »

## Slide 3 — Espaces d'observation / action
Tableau trois lignes (Device : 10 features, actions 2 ; Edge : 10 features, actions 2 ; Cloud : 9 features, actions 2). Référence centralisée : 15 features et 4 actions.
**À l'oral :** « L'approche hiérarchique réduit le besoin d'un état global, mais doit payer un coût de coordination. »

## Slide 4 — Architecture DQN
Q-network 64–64 ReLU ; replay buffer ; target network ; Double DQN ; epsilon-greedy. Diagramme : observation → réseau → scores Q → action.
**À l'oral :** « La mémoire d'expérience stabilise l'entraînement et la cible Double DQN limite la surestimation des Q-values. »

## Slide 5 — Reward
Afficher : `-0.35*latence_norm -0.25*énergie_norm -0.35*miss -0.05*feedback_norm`.
**À l'oral :** « Une tâche en retard reçoit une pénalité explicite ; l'information utile n'est pas gratuite, car chaque message compte en octets et en délai. Les poids sont provisoires et doivent faire l'objet d'une étude de sensibilité. »

## Slide 6 — Feedback et coûts
Trois variantes : `none`, `periodic (tous les 5 tasks)`, `frequent (à chaque task)` ; nombres de messages et octets.
**À l'oral :** « L'objectif de l'ablation est de déterminer si l'amélioration éventuelle des décisions compense le coût du contrôle. »

## Slide 7 — Protocole expérimental
Mêmes tâches/test seed pour 5 politiques ; charges low/medium/high ; réseau good/moderate/congested ; métriques mean/P95 latency, énergie Device, deadline success, overhead.
**À l'oral :** « EDF se compare comme scheduler associé à un placement, ici Greedy, pas comme policy d'offloading indépendante. »

## Slide 8 — Démo et limites
Démonstration terminal : train → evaluate → plot. Afficher `per_task.csv`, `summary.csv`, un graph. Mentionner en grand : `Données synthétiques / simulateur simplifié / raccordement au simulateur partagé nécessaire`.
**À l'oral :** « La partie agents fonctionne comme prototype ; les performances scientifiques seront établies uniquement après intégration au simulateur et à la trace publics de l'Étudiante 1. »
