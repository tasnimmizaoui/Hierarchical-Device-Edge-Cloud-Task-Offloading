> **ARCHIVE DU PROTOTYPE SYNTHÉTIQUE :** non contractuel pour la version SimPy intégrée.

# Partie Étudiante 2 — Apprentissage par renforcement et coordination hiérarchique

**Projet :** Hierarchical Device–Edge–Cloud Task Offloading using Reinforcement Learning for MEC and Cloud/HPC  
**Responsabilité :** conception, implémentation et évaluation des agents RL et du mécanisme de coordination  
**Version :** implémentation de démonstration, à raccorder au simulateur partagé avant conclusions scientifiques.

## 1. Objectif

Le problème de l'**offloading computationnel** consiste à choisir le lieu d'exécution d'une tâche en tenant compte des performances, du réseau, de la consommation du terminal et des échéances. L'approche adoptée distribue la prise de décision entre trois niveaux successifs : Device, Edge et Cloud/HPC. Chaque agent dispose d'une observation locale et choisit une action adaptée à son niveau. Une communication limitée de retour (`feedback`) fournit les informations inter-niveaux explicitement autorisées. Cette architecture s'oppose à une méthode RL centralisée, où une seule politique exploite un état agrégé global.

**Hypothèse d'architecture :** la route obligatoire est Device → Edge → Cloud/HPC. Aucun chemin direct du terminal au cloud n'est autorisé dans l'expérience principale.

## 2. Modélisation RL

### 2.1 États, actions, politiques

Le problème est formulé comme une décision séquentielle partiellement observée par les différents agents. Pour chaque tâche `i`, les agents actifs suivant la route choisie effectuent leurs décisions dans l'ordre Device, Edge, puis Cloud. L'environnement produit une latence, une énergie et un statut d'échéance.

**Agent Device** : observation `s_d` = (travail MI, taille input, temps restant, fraction batterie, capacité locale, attente locale, débit et latence vers Edge, feedback Edge si disponible et indicateur de présence). Actions `A_d={0:local,1:edge}`.

**Agent Edge** : observation `s_e` = (taille et utilisation de queue Edge, capacité, attente, temps restant après transfert, travail, débit et latence vers Cloud, feedback Cloud si disponible et indicateur de présence). Actions `A_e={0:edge,1:cloud}`.

**Agent Cloud/HPC** : observation `s_c` = (résumé queue HPC, utilisation, capacité libre estimée, travail en cours, résumé des deadlines, charge de la tâche et indicateur de complétions récentes (synthétique)). Actions `A_c={0:FIFO,1:EDF}`. Le choix représente le mode de scheduling dans le substitut ; dans la simulation finale il devra effectivement modifier la file de travail.

**Politique centralisée** : observation globale de référence `s_g` avec 15 caractéristiques ; actions `A_g={0:local,1:edge,2:cloud/FIFO,3:cloud/EDF}`. Il s'agit d'une comparaison idéale dotée de plus d'information, donc pas d'une architecture à coût de contrôle identique.

### 2.2 Fonction de récompense

Pour une tâche `i`, nous retenons :

```text
R_i = -[0.35 * min(L_i/D_i, 3) + 0.25 * min(E_i/10, 3)
        + 0.35 * miss_i + 0.05 * min(B_i/64, 3)]
```

- `L_i` : temps total de l'arrivée à la fin de réception du résultat, en secondes.
- `D_i` : deadline relative, en secondes.
- `E_i` : énergie consommée côté Device, en joules ; calcul CPU local + transmission radio + veille selon la route.
- `B_i` : octets des messages de coordination.
- `E_ref=10 J`, `B_ref=64 octets` : constantes de normalisation **choisies pour la démonstration**.
- `(w_L,w_E,w_D,w_C)=(0.35,0.25,0.35,0.05)` : pondérations provisoires à ajuster et justifier par étude de sensibilité.

La récompense finale commune aux agents participants permet l'attribution du crédit après achèvement de la tâche. L'agent Cloud ne reçoit aucune transition pour une tâche exécutée localement ; l'agent Edge n'en reçoit pas non plus si la tâche reste sur le Device.

### 2.3 Temporalité et décisions

Chaque agent applique une politique `a=argmax_a Q(s,a;θ)` en exploitation et une exploration epsilon-greedy à l'entraînement. Le système enchaîne des tâches. La récompense d'une décision sur la tâche `i` est insérée en mémoire lorsque l'agent revoit sa prochaine observation locale. Ce mécanisme définit la transition `(s_t,a_t,r_t,s_{t+1},done)` ; les agents non sollicités n'observent pas de fausse étape de décision.

Le modèle est un **Double DQN indépendant** par niveau : réseau PyTorch de deux couches cachées de 64 neurones ReLU, replay buffer, réseau cible synchronisé périodiquement, Huber loss, Adam et gradient clipping. La valeur cible est :

```text
y_t = r_t + gamma*(1-done)*Q_target(s_next, argmax_a Q_online(s_next,a))
```

avec `γ=0.94`. Les paramètres exacts sont accessibles dans `rl_offloading/dqn.py`. Ce modèle demeure une référence de recherche : la non-stationnarité d'agents indépendants et la pertinence de l'état local doivent être analysées sur un vrai moteur multi-tâches.

## 3. Coordination hiérarchique

La circulation inter-niveaux est limitée aux voisins immédiats : `Cloud → Edge` pour le signal de congestion du cloud ; `Edge → Device` pour le signal de charge de l'edge. Ces deux messages sont générés suivant les modes :

1. `none` : aucun feedback, attributs indisponibles marqués `-1` avec masque de présence `0`.
2. `periodic` : un message par voisin tous les `N=5` arrivées de tâches.
3. `frequent` : messages à chaque arrivée.

Hypothèse de démonstration : 16 octets de payload + 16 octets d'en-tête = 32 octets par message, donc 64 octets par mise à jour complète. Le délai de réception est modélisé par `propagation + bytes/bandwidth` par lien, et imputé à la latence de décision, hypothèse conservatrice. La fréquence et les tailles sont des **valeurs paramétrables**, non des mesures réseau réelles.

Les indicateurs de coordination sont : nombre total de messages, volume transmis et latence ajoutée. En comparant les trois modes sur les mêmes tâches, il est possible de quantifier le coût de communication face au changement de la performance d'offloading.

## 4. Entraînement et validation

Les données de démonstration sont générées par un surrogate. Les variables (MI, bytes, capacité, waiting times, deadlines, capacités réseau, files) proviennent toutes de lois ou intervalles configurés dans `simulator.py` ; **elles ne proviennent pas d'Alibaba Cluster Trace**. Les seeds d'entraînement suivent `seed + 13 × epoch`, la validation fixe `seed + 500000`, le test `901` par défaut. Le meilleur checkpoint de chaque système est conservé selon la récompense moyenne en validation. Aucun résultat du test n'est utilisé pour l'entraînement ou le choix du checkpoint.

Le scénario final demandé doit réutiliser la sélection de tâches et la vraie simulation de l'Étudiante 1. Les transformations de champs trace → MI, taille input/output et deadline doivent être documentées séparément.

## 5. Comparaison expérimentale planifiée

Les politiques à comparer avec la même charge sont : Random, Greedy/Nearest, Greedy+EDF (fixation de la règle de placement), Centralized DQN et Hierarchical DQN. La grille obligatoire comprend : charge faible/moyenne/forte × réseau bon/modéré/congestionné. La hiérarchie est évaluée en modes feedback nul/périodique/fréquent. Les sorties de chaque exécution enregistrent les latences moyenne/P95, le taux d'échéance respectée, l'énergie Device moyenne par tâche soumise, les coûts de coordination et, sous réserves, le throughput.

> **Attention méthodologique :** EDF est une règle de scheduling et non une politique d'offloading autonome. La comparaison `Greedy+EDF` associe explicitement un placement Greedy à un ordre de service EDF.

Une comparaison équitable nécessite mêmes traces, topologies, modèles énergétiques et graines. La vue globale du centralisé constitue un avantage informationnel ; son coût d'observation/communication doit être pris en compte ou identifié comme borne idéale.

## 6. Implémentation et tests effectués

Le code est organisé en modules : génération de contexte et coût `simulator.py` ; DQN et replay `dqn.py` ; wrapper Gymnasium `gym_envs.py` ; entraînement/validation `train.py` ; exécution des politiques/exports `runner.py` ; évaluation `evaluate.py` ; figures `plot_results.py` ; tests `tests/`.

Les tests du coeur du système couvrent reproductibilité, décomposition de la latence, sens des décisions, visibilité du feedback, dimensions des observations, sauvegarde/rechargement d'un modèle et format des résultats. Les tests spécifiques Gymnasium sont prévus ; ils nécessitent son installation dans l'environnement d'exécution. **Les résultats de test livrés concernent uniquement le simulateur substitut**, et non celui de l'Étudiante 1.

## 7. Limites, intégration et travaux restants

Le simulateur substitut utilise un snapshot indépendant de files à chaque tâche : il n'assure pas un calendrier causal global de ressources partagées. Son EDF estime la position de la tâche entrante, sans traiter les conséquences sur les autres tâches. Il n'exécute aucun vrai job HPC. Les graphes et CSV accompagnant cette version illustrent le fonctionnement du code, **pas la qualité scientifique de la méthode sur un réseau MEC ou un cluster HPC**.

Avant conclusion finale, intégrer les vraies files à événements discrets, des paramètres traçables, le traitement des ressources partagées, la trace prétraitée et les baselines de l'Étudiante 1. Réentraîner et refaire tous les résultats sur le jeu de test bloqué. Vérifier la sensibilité à la pondération des récompenses, à la fréquence de feedback et aux seeds ; produire des intervalles d'incertitude sur plusieurs runs.

## 8. Conclusion provisoire

Une structure multi-agent hiérarchique a été spécifiée et implémentée dans un banc de démonstration exécutable, avec décision Device/Edge/Cloud, retour de congestion et référence centralisée. Cette étape permet de vérifier les interfaces, le cycle d'entraînement, la sauvegarde des modèles et la collecte uniforme des métriques. **Aucune conclusion de supériorité de la politique hiérarchique n'est justifiée avant la validation avec le simulateur final et les traces prévues.**

## 9. Références

[1] Mnih et al. (2015), *Human-level control through deep reinforcement learning*, Nature 518, 529–533, https://doi.org/10.1038/nature14236.

[2] van Hasselt, Guez, Silver (2016), *Deep Reinforcement Learning with Double Q-learning*, AAAI, https://arxiv.org/abs/1509.06461.

[3] Gymnasium, *Env API*, https://gymnasium.farama.org/main/api/env/.

[4] PyTorch, *Reinforcement Learning (DQN) Tutorial*, https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.
