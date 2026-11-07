"""Recreate report figures from the documented final-matched summary CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
SUMMARY = ROOT / "docs" / "experiments" / "final-matched" / "summary.csv"


def plot_comparison() -> None:
    summary = pd.read_csv(SUMMARY)
    policies = [
        "random",
        "greedy_fifo",
        "greedy_edf",
        "centralized_dqn",
        "hierarchical_dqn",
        "hierarchical_double_dqn",
    ]
    labels = ["Random", "G/FIFO", "G/EDF", "Centralized\nDQN",
              "Hierarchical\nDQN", "Hierarchical\nDouble"]
    values = {}
    for metric in ("deadline_success_rate", "p95_latency_s", "mean_device_energy_j"):
        rows = summary[summary.metric == metric].set_index("policy")
        values[metric] = [rows.loc[p, "mean"] for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    colors = ["#7f7f7f", "#70ad47", "#70ad47", "#4472c4", "#ed7d31", "#a64d79"]
    for axis, metric, ylabel, scale in zip(
        axes,
        ("deadline_success_rate", "p95_latency_s", "mean_device_energy_j"),
        ("Deadline success (%)", "P95 latency (s)", "Device energy (J/task)"),
        (100, 1, 1),
    ):
        axis.bar(range(len(labels)), [v * scale for v in values[metric]], color=colors)
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "final_matched_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def topology_figure() -> None:
    fig, axis = plt.subplots(figsize=(7.2, 2.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.axis("off")
    edge_x = [3.0, 5.0, 7.0]
    for index, x in enumerate(edge_x, start=1):
        axis.add_patch(FancyBboxPatch(
            (x - 0.55, 1.65), 1.1, 0.55, boxstyle="round,pad=0.03",
            facecolor="#d9ead3", edgecolor="#17365d",
        ))
        axis.text(x, 1.92, f"Edge {index}", ha="center", va="center", fontsize=8)
    for i in range(24):
        x = edge_x[i % 3] + ((i // 3) - 3.5) * 0.18
        y = 3.1 + (0.18 if (i // 3) % 2 else 0)
        axis.add_patch(FancyBboxPatch(
            (x - 0.10, y - 0.10), 0.20, 0.20, boxstyle="round,pad=0.01",
            facecolor="#d9eaf7", edgecolor="#17365d",
        ))
        axis.text(x, y, str(i + 1), ha="center", va="center", fontsize=5)
        axis.add_patch(FancyArrowPatch(
            (x, y - 0.12), (edge_x[i % 3], 2.22),
            arrowstyle="-", linewidth=0.35, color="#888888",
        ))
    axis.add_patch(FancyBboxPatch(
        (4.15, 0.35), 1.7, 0.55, boxstyle="round,pad=0.03",
        facecolor="#fce5cd", edgecolor="#17365d",
    ))
    axis.text(5.0, 0.62, "Shared Cloud (1 worker)", ha="center", va="center", fontsize=8)
    for x in edge_x:
        axis.add_patch(FancyArrowPatch(
            (x, 1.62), (5.0, 0.92), arrowstyle="-|>", mutation_scale=10,
            linewidth=0.8, color="#555555",
        ))
    axis.set_title("Reference RL topology: 24 devices, 3 edges, 1 shared cloud", fontsize=10)
    fig.savefig(OUT / "topology_24_devices.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_comparison()
    topology_figure()
