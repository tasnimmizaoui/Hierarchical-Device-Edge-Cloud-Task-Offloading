"""Interactive controls and results for the Streamlit demonstration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from scripts.run_baselines import build_policies

from mec_offloading.evaluation.runner import aggregate_results
from mec_offloading.rl.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "artifacts/models/feedback-ablation"
FEEDBACK_NAMES = {
    "frequent": "À chaque décision",
    "periodic": "Périodique",
    "event": "Sur événement",
    "none": "Sans feedback",
}
SOURCES = ("Simulation synthétique", "Trace Alibaba et modèles RL")
SCENARIO_NAMES = {
    "nominal": "Charge normale",
    "low_load": "Arrivées espacées",
    "high_load": "Arrivées rapprochées",
    "congested_network": "Réseau ralenti",
}
POLICY_NAMES = {
    "local": "Tout sur l’appareil",
    "edge_execute": "Tout sur l’edge",
    "cloud_edf": "Cloud · priorité à l’échéance",
    "cloud_fifo": "Cloud · ordre d’arrivée",
    "greedy_edge": "Greedy · appareil ou edge",
    "greedy_fifo": "Greedy global · FIFO",
    "greedy_edf": "Greedy global · EDF",
    "greedy_delayed": "Greedy · informations différées",
    "random": "Aléatoire",
    "centralized_rl": "RL centralisé",
}
METRICS = {
    "deadline_success_rate": "Échéances respectées (%)",
    "p95_latency_s": "Latence P95 (s)",
    "mean_latency_s": "Latence moyenne (s)",
    "mean_device_energy_j": "Énergie moyenne de l’appareil (J)",
    "communication_bytes": "Données transférées (octets)",
    "feedback_messages": "Messages de coordination",
}


def run_synthetic(count, seed, scenario_name, policies):
    scenario = SCENARIOS[scenario_name]
    tasks = scenario.tasks(count, seed)
    deadlines = {task.task_id: task.deadline_s for task in tasks}
    summary, details = [], []
    for policy in build_policies(seed):
        if policy.name not in policies:
            continue
        results = scenario.backend().run_episode(
            tasks, policy.device, policy.edge, policy.cloud, seed=seed
        )
        summary.append({
            "scenario": scenario_name,
            **aggregate_results(policy.name, results, seed=seed),
        })
        details.extend({
            **asdict(result),
            "policy": policy.name,
            "scenario": scenario_name,
            "seed": seed,
            "route": "->".join(tier.value for tier in result.route),
            "deadline_s": deadlines[result.task_id],
        } for result in results)
    return pd.DataFrame(summary), pd.DataFrame(details)


def policy_name(value):
    if value.startswith("hierarchical_"):
        _, _, rule = value.partition("_fixed_")
        if rule == "shortest_processing_time":
            rule = "spt"
        return f"RL hiérarchique · cloud {rule.upper()}" if rule else "RL hiérarchique"
    return POLICY_NAMES.get(value, value)


def run_alibaba(count, seed, ablation, model, metadata):
    # Keep the same compatibility checks and trace split as the CLI evaluation.
    results_root = ROOT / "results"
    results_root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="demo-", dir=results_root) as directory:
        output = Path(directory)
        command = [
            sys.executable, str(ROOT / "scripts/compare_integrated_policies.py"),
            "--source", "alibaba2017", "--deadline-policy", metadata["deadline_policy"],
            "--feedback", metadata["feedback"], "--tasks", str(count), "--seeds", str(seed),
            "--feedback-frequency", str(metadata.get("feedback_frequency", 5)),
            "--event-thresholds", str(metadata.get("event_threshold", 0.25)),
            "--max-trace-tasks", "5000", "--models", str(model),
            "--output", str(output),
        ]
        if ablation:
            command.append("--cloud-rule-ablation")
        process = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300, check=False,
        )
        if process.returncode:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip())
        return (
            pd.read_csv(output / "aggregate_metrics.csv"),
            pd.read_csv(output / "task_results.csv"),
        )


def display_results(summary, details, source):
    st.header("2. Comparer les stratégies")
    scenarios = summary["scenario"].unique()
    scenario = st.selectbox(
        "Scénario affiché", scenarios,
        format_func=lambda value: SCENARIO_NAMES.get(
            value, value.replace("alibaba2017_test", "Alibaba · test")
        ),
        disabled=len(scenarios) == 1,
    )
    comparison = summary[summary["scenario"] == scenario].copy()
    first = comparison.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Requêtes par stratégie", int(first["task_count"]))
    cols[1].metric("Stratégies simulées", len(comparison))
    cols[2].metric("Graine utilisée", int(first["seed"]))
    cols[3].metric("Source", "Alibaba" if source == SOURCES[1] else "Synthétique")
    comparison["Stratégie"] = comparison["policy"].map(policy_name)
    comparison["deadline_success_rate"] *= 100
    chart = st.selectbox("Indicateur du graphique", list(METRICS), format_func=METRICS.get)
    st.bar_chart(comparison.set_index("Stratégie")[chart].rename(METRICS[chart]))
    st.dataframe(
        comparison.set_index("Stratégie")[list(METRICS)].rename(columns=METRICS).round(3),
        width="stretch",
    )
    st.caption(
        "Le P95 est le temps sous lequel environ 95 % des requêtes se terminent. "
        "Les stratégies du scénario reçoivent les mêmes requêtes."
    )

    st.header("3. Suivre une requête")
    subset = details[details["scenario"] == scenario]
    left, right = st.columns(2)
    policy = left.selectbox("Stratégie à explorer", subset["policy"].unique(),
                            format_func=policy_name)
    tasks = subset[subset["policy"] == policy].set_index("task_id")
    routes = tasks["route"].value_counts().rename_axis("Trajet").rename("Requêtes")
    with st.expander("Répartition des trajets"):
        st.bar_chart(routes)
    task_id = right.selectbox(
        "Requête", tasks.index.tolist(),
        format_func=lambda value: f"{value} · {tasks.at[value, 'route'].replace('->', ' → ')}",
    )
    task = tasks.loc[task_id]
    st.subheader(task["route"].replace("->", " → ").replace("device", "Appareil"))
    cols = st.columns(4)
    cols[0].metric("Latence", f"{task['end_to_end_latency_s']:.3f} s")
    cols[1].metric("Attente en file", f"{task['queue_wait_s']:.3f} s")
    cols[2].metric("Énergie appareil", f"{task['device_energy_j']:.3f} J")
    cols[3].metric("Échéance", "Respectée" if task["deadline_met"] else "Dépassée")
    if "deadline_s" in task:
        st.caption(f"Délai autorisé depuis l’arrivée · {task['deadline_s']:.3f} s")
    if pd.notna(task["failure_reason"]) and task["failure_reason"]:
        st.warning(f"Motif enregistré · {task['failure_reason']}")
    st.caption(
        f"Arrivée à {task['arrival_time_s']:.3f} s · fin à "
        f"{task['completion_time_s']:.3f} s · "
        f"{int(task['communication_bytes']):,} octets transférés"
    )
    with st.expander("Exporter les résultats"):
        st.download_button("Télécharger la comparaison", summary.to_csv(index=False),
                           "comparaison.csv", "text/csv")
        st.download_button("Télécharger les requêtes", details.to_csv(index=False),
                           "requetes.csv", "text/csv")


def show_demo():
    st.title("Simulateur d’offloading")
    st.markdown("**Appareil → Edge → Cloud**")
    st.write("Règle la charge, lance l’expérience, puis observe où les requêtes sont exécutées.")
    st.header("1. Configurer la simulation")
    source = st.radio("Source des requêtes", SOURCES, horizontal=True)
    alibaba = source == SOURCES[1]
    if alibaba:
        left, right = st.columns(2)
        algorithm = left.selectbox("Modèle préentraîné", ("double-dqn", "dqn"),
                                   format_func=lambda value: value.upper())
        feedback = right.selectbox("Coordination entre agents", list(FEEDBACK_NAMES),
                                   format_func=FEEDBACK_NAMES.get)
        model = MODELS / f"fb-{feedback}-threshold-0.25-{algorithm}-seed-42-next-decision"
        metadata_path = model / "metadata.json"
        required = [
            metadata_path, *(model / f"{agent}.pt"
                             for agent in ("device", "edge", "cloud", "centralized")),
            ROOT / "data/raw/batch_task.csv", ROOT / "data/raw/batch_instance.csv",
            ROOT / "configs/alibaba-multi-topology.yaml",
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            st.warning("Mode Alibaba indisponible. Fichiers manquants · " + ", ".join(missing))
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        maximum = metadata["workload"]["test_tasks"]
        st.caption(
            "24 appareils, 3 edges, 1 cloud partagé. Comparaison des modèles préentraînés "
            "et des heuristiques sur le jeu de test."
        )
    else:
        maximum = 5000
        st.caption("1 appareil, 1 edge, 1 cloud. Charge générée à partir de la graine choisie.")

    with st.form(f"simulation_{alibaba}"):
        left, right = st.columns(2)
        count = left.number_input("Nombre de requêtes", 1, maximum, min(100, maximum),
                                  step=1, help="Nombre évalué pour chaque stratégie et scénario.")
        seed = right.number_input("Graine de simulation", 0, 2**32 - 1, 10001, step=1,
                                  help="Réutilise la même graine pour reproduire la charge.")
        if alibaba:
            ablation = st.checkbox("Comparer aussi les règles cloud EDF, SPT et FIFO")
        else:
            scenario = st.selectbox("Conditions de simulation", list(SCENARIOS),
                                    format_func=SCENARIO_NAMES.get)
            policies = st.multiselect(
                "Stratégies à comparer", [policy.name for policy in build_policies(0)],
                default=["local", "edge_execute", "cloud_edf", "greedy_edge"],
                format_func=policy_name,
            )
        submitted = st.form_submit_button("Lancer la simulation", type="primary")

    if submitted:
        st.session_state.pop("demo_result", None)
        if not alibaba and not policies:
            st.warning("Sélectionne au moins une stratégie.")
        else:
            message = "Lecture de la trace et évaluation RL…" if alibaba else "Simulation SimPy…"
            try:
                with st.spinner(message, show_time=True):
                    result = (run_alibaba(count, seed, ablation, model, metadata) if alibaba
                              else run_synthetic(count, seed, scenario, policies))
            except subprocess.TimeoutExpired:
                st.error("L’évaluation a dépassé la limite de cinq minutes.")
            except (OSError, RuntimeError, ValueError) as error:
                st.error("La simulation a échoué.")
                with st.expander("Détails de l’erreur"):
                    st.code(str(error))
            else:
                st.session_state["demo_result"] = (*result, source, model if alibaba else None)
                st.success("Simulation terminée.")

    if "demo_result" in st.session_state:
        summary, details, result_source, result_model = st.session_state["demo_result"]
        if result_source == source and (not alibaba or result_model == model):
            display_results(summary, details, result_source)
