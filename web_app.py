"""Interactive *read-only* dashboard for reproducible SimPy CSV experiments.

Run: streamlit run web_app.py
No metrics are generated, interpolated or manually fabricated by this UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MEC · Device / Edge / Cloud", layout="wide", page_icon="☁️")
st.title("☁️ Offloading Device → Edge → Cloud/HPC")
st.caption("Projet 05 · Étudiante 2 / intégration RL · Dashboard de développement")

result_dir = Path(st.sidebar.text_input("Dossier des évaluations", "results/integrated-comparison"))
model_dir = Path(st.sidebar.text_input("Dossier des entraînements", "artifacts/models/integrated"))

metadata_file = result_dir / "metadata.json"
if metadata_file.is_file():
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    st.warning("Données de développement SYNTHÉTIQUES — topo 1 device / 1 edge / 1 cloud. "
               "Les résultats ne valident pas encore une trace Alibaba/Google.")
    st.caption(f"Moteur : {metadata.get('simulator', 'SimPy')} · "
               f"Seeds test : {metadata.get('test_seeds', 'non précisées')} · "
               f"Scénarios : {metadata.get('scenarios', ['nominal'])}")
else:
    st.info("Aucune évaluation trouvée pour ce dossier. Exécuter l'entraînement "
            "puis la comparaison (commandes dans README).")

summary_file = result_dir / "aggregate_metrics.csv"
detail_file = result_dir / "task_results.csv"
summary = pd.read_csv(summary_file) if summary_file.is_file() else pd.DataFrame()
details = pd.read_csv(detail_file) if detail_file.is_file() else pd.DataFrame()
training_file = model_dir / "training_log.csv"
training = pd.read_csv(training_file) if training_file.is_file() else pd.DataFrame()
for table in (summary, details):
    if not table.empty:
        if "scenario" not in table:
            table["scenario"] = "nominal"
        if "feedback_mode" not in table:
            table["feedback_mode"] = "n/a"

overview, training_tab, coordination, explorer = st.tabs([
    "📊 Comparaison", "🧠 Entraînement RL", "📡 Coordination", "🔍 Parcours des tâches",
])

with overview:
    if summary.empty:
        st.info("Les courbes apparaîtront après une expérience enregistrée.")
    else:
        scenarios = sorted(summary["scenario"].dropna().unique().tolist())
        scenario = st.selectbox("Scénario de comparaison", scenarios, key="overview_scenario")
        data = summary[summary["scenario"] == scenario]
        policies = sorted(data["policy"].dropna().unique().tolist())
        chosen = st.multiselect("Politiques", policies, default=policies,
                                key="overview_policies")
        data = data[data["policy"].isin(chosen)]
        if data.empty:
            st.info("Sélectionner au moins une politique.")
        else:
            means = data.groupby("policy").agg(
                latency_s=("mean_latency_s", "mean"),
                p95_latency_s=("p95_latency_s", "mean"),
                success_rate=("deadline_success_rate", "mean"),
                energy_j=("mean_device_energy_j", "mean"),
                feedback_bytes=("feedback_bytes", "mean"),
                feedback_messages=("feedback_messages", "mean"),
            )
            a, b, c, d = st.columns(4)
            a.metric("Politiques", len(means))
            b.metric("Seeds de test", data["seed"].nunique())
            c.metric("Scénario", scenario)
            d.metric("Nombre de tâches/seed", int(data["task_count"].max()))
            left, right = st.columns(2)
            with left:
                st.markdown("**Latence moyenne (s) — plus bas = mieux**")
                st.bar_chart(means["latency_s"])
                st.markdown("**Énergie device par tâche (J) — plus bas = mieux**")
                st.bar_chart(means["energy_j"])
            with right:
                st.markdown("**Deadlines respectées (%) — plus haut = mieux**")
                st.bar_chart(means["success_rate"] * 100)
                st.markdown("**Latence P95 (s)**")
                st.bar_chart(means["p95_latency_s"])
            st.markdown("**Mesures par seed, sans moyenne inter-scénarios**")
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.download_button("Exporter le résumé CSV", data.to_csv(index=False),
                               "summary_filtered.csv", "text/csv")

with training_tab:
    st.markdown("**Suivi des modèles RL — données enregistrées, non simulées par l'interface**")
    if training.empty:
        st.info("Lancer `python scripts/train_integrated_rl.py` pour afficher "
                "récompenses, pertes, epsilon et actions.")
    else:
        train = training[training["split"] == "train"].copy()
        valid = training[training["split"] == "validation"].copy()
        tiers = sorted(train["tier"].unique().tolist())
        chosen_tier = st.selectbox("Agent", tiers, key="training_tier")
        item = train[train["tier"] == chosen_tier].sort_values("episode")
        if not item.empty:
            left, right = st.columns(2)
            with left:
                st.markdown("**Reward / tâche**")
                st.line_chart(item.set_index("episode")["team_task_reward"])
                st.markdown("**Loss Double-DQN (uniquement si mises à jour)**")
                losses = item.dropna(subset=["loss"])
                if losses.empty:
                    st.info("Pas encore assez de transitions pour un minibatch.")
                else:
                    st.line_chart(losses.set_index("episode")["loss"])
            with right:
                st.markdown("**Exploration epsilon**")
                st.line_chart(item.set_index("episode")["epsilon"])
                st.markdown("**Fréquence des actions / épisode**")
                st.bar_chart(item.set_index("episode")[["action_0", "action_1", "action_2"]])
        if not valid.empty:
            st.markdown("**Validation indépendante (greedy, seeds hors entraînement)**")
            st.line_chart(valid.pivot_table(
                index="episode", columns="tier", values="team_task_reward", aggfunc="mean",
            ))
            st.dataframe(valid, use_container_width=True, hide_index=True)
        model_metadata = model_dir / "metadata.json"
        if model_metadata.exists():
            config = json.loads(model_metadata.read_text(encoding="utf-8"))
            st.caption("Meilleurs checkpoints sélectionnés sur validation; "
                       f"seeds = {config.get('validation_seeds', [])}. "
                       "Fichiers finaux non sélectionnés : last/.")

with coordination:
    if summary.empty:
        st.info("Les statistiques de communication apparaîtront après comparaison.")
    else:
        scenario_coord = st.selectbox(
            "Scénario", sorted(summary["scenario"].unique()), key="coord_scenario",
        )
        coord = summary[summary["scenario"] == scenario_coord]
        hierarchy = coord[coord["policy"].str.startswith("hierarchical_")].copy()
        if hierarchy.empty:
            st.info("Aucune ablation hiérarchique trouvée.")
        else:
            mean = hierarchy.groupby("feedback_mode").agg(
                bytes_sent=("feedback_bytes", "mean"),
                messages=("feedback_messages", "mean"),
                reward_proxy_latency=("mean_latency_s", "mean"),
                deadline_rate=("deadline_success_rate", "mean"),
                modeled_delay=("mean_feedback_modeled_delay_s", "mean"),
            )
            left, right = st.columns(2)
            with left:
                st.markdown("**Messages envoyés par expérience**")
                st.bar_chart(mean["messages"])
                st.markdown("**Bytes de contrôle transmis**")
                st.bar_chart(mean["bytes_sent"])
            with right:
                st.markdown("**Latence de bout en bout (s)**")
                st.bar_chart(mean["reward_proxy_latency"])
                st.markdown("**Respect des deadlines (%)**")
                st.bar_chart(mean["deadline_rate"] * 100)
            st.dataframe(mean, use_container_width=True)
            st.caption("Messages à réception différée au moment des décisions; "
                       "pas de contention du trafic de contrôle avec les tâches. "
                       "Le délai de feedback n'est pas ajouté automatiquement à la durée d'une tâche.")

with explorer:
    if details.empty:
        st.info("Les parcours sont visibles une fois les CSV produits.")
    else:
        sc = st.selectbox("Scénario", sorted(details["scenario"].unique()), key="trace_scenario")
        subset = details[details["scenario"] == sc]
        p = st.selectbox("Politique", sorted(subset["policy"].unique()), key="trace_policy")
        subset = subset[subset["policy"] == p]
        seed = st.selectbox("Seed", sorted(subset["seed"].unique()), key="trace_seed")
        subset = subset[subset["seed"] == seed].copy()
        subset["destination"] = subset["route"].str.split("->").str[-1]
        st.markdown("**Répartition des destinations finales**")
        st.bar_chart(subset["destination"].value_counts())
        st.dataframe(subset, use_container_width=True, hide_index=True)
        st.download_button("Exporter les parcours CSV", subset.to_csv(index=False),
                           f"tasks_{sc}_{p}_{seed}.csv", "text/csv")

st.divider()
st.caption("Étudiante 2 — prototype V2. Les performances doivent être réévaluées "
           "sur les traces et topologies de l'étudiante 1 avant le rapport final.")
