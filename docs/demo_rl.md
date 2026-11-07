# Démonstration fonctionnelle du projet MEC

Prévoir cinq à six minutes. Préparer deux fenêtres dans le dossier du dépôt,
un terminal PowerShell et le tableau de bord Streamlit. Les checkpoints et
les CSV sont des artefacts locaux. Vérifier leur présence sur la machine de
soutenance avant la séance.

## 1. Exécution en direct (1 minute)

```powershell
.venv/Scripts/python.exe scripts/run_simulation_smoke.py
```

Montrer les routes device, device–edge et device–edge–cloud avec les mêmes
trois tâches synthétiques. Lire une ligne de sortie et expliquer les mesures
de latence, énergie et octets. Cette vérification exécute réellement SimPy.
Pour montrer l'inférence à trois agents en direct, préparer ce rejeu de
validation technique dans un deuxième terminal. L'analyse complète de
l'article utilise les 579 tâches.

```powershell
.venv/Scripts/python.exe scripts/compare_integrated_policies.py --source alibaba2017 --deadline-policy best-tier-calibrated --feedback frequent --tasks 100 --max-trace-tasks 5000 --seeds 10001 --models artifacts/models/cloud-pilot-learned/fb-frequent-seed-42-terminal --output results/demo-live
```

Le chargement de la trace peut prendre du temps même avec 100 tâches
évaluées. Lancer cette commande avant d'ouvrir les onglets du tableau de bord
et revenir à sa sortie à la fin. Le tirage réduit sert à la démo.

## 2. Tableau de bord (3 minutes)

```powershell
.venv/Scripts/python.exe -m streamlit run web_app.py
```

Dans la barre latérale, vérifier les chemins du pilote à trois agents.

| Champ | Valeur à saisir |
|---|---|
| Dossier de comparaison | `results/cloud-pilot-learned/fb-frequent-seed-42-terminal` |
| Dossier du modèle | `artifacts/models/cloud-pilot-learned/fb-frequent-seed-42-terminal` |

1. **Comparaison.** Choisir `alibaba2017_test`. Montrer la politique
   hiérarchique, le DQN centralisé et le greedy. Expliquer la différence
   d'information entre le greedy global et le greedy retardé.
2. **Entraînement RL.** Choisir l'agent `cloud`. Montrer les actions
   EDF/SPT/FIFO et expliquer la différence entre exploration et évaluation.
3. **Parcours des tâches.** Choisir `hierarchical_frequent` et la graine
   `10001`. Suivre une tâche jusqu'au cloud. Montrer sa deadline, son
   énergie et sa latence à partir du résultat enregistré.
4. **Coordination.** Ouvrir `results/feedback-ablation/fb-event-seed-42-terminal`
   dans le champ de comparaison pour montrer le mode événementiel. Pour
   comparer les quatre modes sur les trois graines, afficher le graphique
   du rapport. Le tableau de bord lit un seul dossier à la fois.

L'interface lit les CSV et les journaux sauvegardés. À la fin du rejeu,
renseigner `results/demo-live` pour montrer les résultats produits devant
le jury. Garder les résultats complets ouverts dans le PDF si le rejeu
prend plus de temps.

## 3. Résultat à expliquer (1 minute)

Ouvrir le tableau de l'ablation cloud à poids amont gelés. Sur la
référence, le P95 passe de 19,87 s avec EDF à 15,64 s avec le DQN.
SPT fixe atteint 15,61 s. Le graphique des actions montre que deux
checkpoints choisissent toujours SPT quand plusieurs tâches attendent
et le troisième toujours FIFO. Terminer par le graphique coût–deadline
du feedback événementiel. La baisse de 69 % concerne les messages de
coordination RL. Le greedy retardé reste devant en taux de succès brut.

Préparer la réponse à trois questions du jury.

- **D'où viennent les données ?** Arrivées et calcul dérivés du trace
  Alibaba. Charges réseau, deadlines et topologie sont paramétrées.
- **Où intervient le cloud ?** Il choisit une règle d'ordre de service
  pour une file partagée. Le modèle possède un seul exécuteur.
- **Comment isoler son effet ?** Rejouer EDF et DQN avec le même
  checkpoint amont, les mêmes tâches et les mêmes perturbations.
