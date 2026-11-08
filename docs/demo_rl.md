# Démonstration interactive du projet MEC

Depuis la racine du dépôt, ouvrir l’application une fois.

```powershell
.\.venv\Scripts\python.exe -m streamlit run web_app.py
```

Ouvrir http://localhost:8501. Les réglages, le lancement et l’exploration se font ensuite dans l’application.

## Montrer l’effet de la charge

1. Dans **Simuler**, choisir **Simulation synthétique**, saisir **20** requêtes et garder la charge normale. Lancer la simulation.
2. Comparer les échéances respectées et la latence P95. Dans **Suivre une requête**, passer de **Tout sur l’appareil** à **Tout sur l’edge**, puis au cloud EDF.
3. Garder la même graine, passer à **100** requêtes et choisir **Arrivées rapprochées**. Relancer et regarder comment l’attente et les échéances évoluent. Pour isoler l’effet d’un seul réglage, faire deux essais en ne modifiant que ce réglage.

## Comparer les modèles RL

1. Choisir **Trace Alibaba et modèles RL**, choisir **Double DQN** et la coordination **À chaque décision**, saisir **100** requêtes et lancer la simulation. Le chargement des traces locales peut prendre du temps. Les modèles sont déjà entraînés.
2. Dans **Scénario affiché**, garder **Alibaba · test** pour la topologie de référence. Comparer le RL hiérarchique, le RL centralisé et les heuristiques.
3. Pour isoler la décision du cloud, cocher **Comparer aussi les règles cloud EDF, SPT et FIFO** et relancer. Les règles fixes utilisent les mêmes modèles appareil et edge.
4. Exporter les mesures depuis **Exporter les résultats**. L’espace **Résultats archivés** donne accès aux analyses et journaux enregistrés.

Le nombre saisi s’applique à chaque stratégie et scénario. Les arrivées et le calcul du mode Alibaba dérivent de la trace. Le réseau et les échéances reposent sur les hypothèses du simulateur.
