"""Rebuild the figures embedded in docs/rl_experiment_report.md."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")


POLICIES = (
    ("greedy_edf", "Greedy + EDF"),
    ("centralized_rl", "Centralized DQN"),
    ("hierarchical_none", "Hierarchical / no feedback"),
    ("hierarchical_periodic", "Hierarchical / periodic"),
    ("hierarchical_frequent", "Hierarchical / frequent"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _box(axis, xy, width, height, title, body, color):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color,
        edgecolor="#25364a",
        linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height * 0.68, title, ha="center",
              va="center", weight="bold", fontsize=10)
    axis.text(x + width / 2, y + height * 0.34, body, ha="center",
              va="center", fontsize=8, wrap=True)


def _arrow(axis, start, end, label=None, color="#34495e", style="-|>"):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12, linewidth=1.2, color=color
    ))
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025,
                  label, ha="center", va="bottom", fontsize=7, color=color)


def architecture_figure(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 6.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(
        "Integrated task-offloading policies and Double-DQN learning path",
        fontsize=15,
        weight="bold",
        pad=14,
    )

    _box(axis, (0.03, 0.70), 0.19, 0.18, "Alibaba 2017 tasks",
         "Arrival + representative-instance MI\nMEC fields are explicit assumptions",
         "#eaf2f8")
    _box(axis, (0.30, 0.70), 0.19, 0.18, "Device DQN",
         "11 features · 2 actions\nlocal / offload to edge", "#d5f5e3")
    _box(axis, (0.56, 0.70), 0.19, 0.18, "Edge DQN",
         "10 features · 2 actions\nexecute / forward to cloud", "#d5f5e3")
    _box(axis, (0.81, 0.70), 0.16, 0.18, "Cloud scheduler",
         "Fixed EDF in reported runs\n(no learned cloud checkpoint)", "#fcf3cf")
    _arrow(axis, (0.22, 0.79), (0.30, 0.79), "tasks")
    _arrow(axis, (0.49, 0.79), (0.56, 0.79), "offload")
    _arrow(axis, (0.75, 0.79), (0.81, 0.79), "forward")
    _arrow(axis, (0.38, 0.58), (0.38, 0.66), "delayed feedback", color="#2874a6")
    _arrow(axis, (0.65, 0.58), (0.65, 0.66), "delayed feedback", color="#2874a6")
    _box(axis, (0.29, 0.43), 0.44, 0.15, "SimPy shared event timeline",
         "3 devices · 2 edges · one cloud worker\none execution slot per device/edge",
         "#ebdef0")
    _box(axis, (0.03, 0.43), 0.19, 0.15, "Centralized DQN",
         "16 features · 3 actions\noracle full-state reference", "#d6eaf8")
    _arrow(axis, (0.22, 0.50), (0.29, 0.50), "reference")
    _box(axis, (0.29, 0.12), 0.44, 0.17, "Episode outcomes → shared task reward",
         "Latency + energy + deadline miss + coordination\nfeasibility-aware miss term",
         "#fadbd8")
    _arrow(axis, (0.51, 0.43), (0.51, 0.29), "task result")
    _box(axis, (0.78, 0.13), 0.19, 0.15, "Replay + updates",
         "One-step terminal credit\nDouble-DQN · target network", "#f5eef8")
    _arrow(axis, (0.73, 0.20), (0.78, 0.20), "reward")
    _arrow(axis, (0.97, 0.20), (0.99, 0.20), color="#7d3c98", style="-")
    _arrow(axis, (0.99, 0.20), (0.99, 0.93), color="#7d3c98", style="-")
    _arrow(axis, (0.99, 0.93), (0.40, 0.93),
           "update acting policy weights", color="#7d3c98", style="-")
    _arrow(axis, (0.40, 0.93), (0.40, 0.88), color="#7d3c98")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def baseline_figure(rows: list[dict[str, str]], output: Path) -> None:
    data = {
        row["policy"]: row for row in rows
        if row["scenario"] == "alibaba2017_test"
    }
    policies = [(key, label) for key, label in POLICIES if key in data]
    labels = [label for _, label in policies]
    values = {
        "Raw deadline success (%)": [
            100 * float(data[key]["deadline_success_rate"]) for key, _ in policies
        ],
        "Feasible-task success (%)": [
            100 * float(data[key]["feasible_deadline_success_rate"])
            for key, _ in policies
        ],
        "P95 latency (s)": [
            float(data[key]["p95_latency_s"]) for key, _ in policies
        ],
        "Total communication (GB)": [
            (
                float(data[key]["communication_bytes"])
                + float(data[key]["feedback_bytes"])
            ) / 1e9
            for key, _ in policies
        ],
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#718096", "#3973ac", "#73a9c2", "#2e8b57", "#88b04b"]
    for axis, (title, scores) in zip(axes.flat, values.items(), strict=True):
        axis.barh(labels, scores, color=colors[:len(labels)])
        axis.set_title(title, weight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()
        for index, value in enumerate(scores):
            axis.text(value, index, f" {value:.2f}", va="center", fontsize=8)
    fig.suptitle("V8 held-out test results — reference topology, 579 tasks",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def topology_figures(
    reference: list[dict[str, str]],
    robust: list[dict[str, str]],
    output: Path,
) -> None:
    variants = list(dict.fromkeys(row["variant"] for row in reference))
    colors = {"v8 hierarchical": "#c0392b", "v9 hierarchical": "#2874a6",
              "Greedy EDF": "#555555"}
    series = {}
    for label, rows, policy in (
        ("v8 hierarchical", reference, "hierarchical_periodic"),
        ("v9 hierarchical", robust, "hierarchical_periodic"),
        ("Greedy EDF", robust, "greedy_edf"),
    ):
        selected = {
            row["variant"]: row for row in rows if row["policy"] == policy
        }
        series[label] = selected

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for label, rows in series.items():
        axes[0].plot(
            variants,
            [100 * float(rows[name]["deadline_success_rate"]) for name in variants],
            marker="o",
            linewidth=1.8,
            label=label,
            color=colors[label],
        )
        axes[1].plot(
            variants,
            [100 * float(rows[name]["feasible_deadline_success_rate"])
             for name in variants],
            marker="o",
            linewidth=1.8,
            label=label,
            color=colors[label],
        )
    axes[0].set_title("Raw deadline success", weight="bold")
    axes[1].set_title("Deadline success among tasks feasible in that variant", weight="bold")
    for axis in axes:
        axis.set_ylabel("Success (%)")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, loc="lower left")
    axes[1].tick_params(axis="x", rotation=35)
    fig.suptitle("Held-out topology robustness — fixed deadlines, same 579 tasks",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    figure, axis = plt.subplots(figsize=(13, 5))
    for label, rows, policy, color in (
        ("v8 hierarchical / periodic", reference, "hierarchical_periodic", "#c0392b"),
        ("v9 hierarchical / periodic", robust, "hierarchical_periodic", "#2874a6"),
        ("Greedy EDF", robust, "greedy_edf", "#555555"),
    ):
        selected = {row["variant"]: row for row in rows if row["policy"] == policy}
        axis.plot(
            variants,
            [float(selected[name]["p95_latency_s"]) for name in variants],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
    axis.set_ylabel("P95 latency (seconds)")
    axis.set_title("Tail latency under topology shift", weight="bold")
    axis.grid(alpha=0.25)
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output.with_name("topology_p95_latency.png"),
                   dpi=180, bbox_inches="tight")
    plt.close(figure)


def training_figure(v8_log: list[dict[str, str]], v9_log: list[dict[str, str]],
                    output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, rows, color in (
        ("V8: baseline validation", v8_log, "#c0392b"),
        ("V9: multi-topology validation", v9_log, "#2874a6"),
    ):
        validation = [
            row for row in rows
            if row["split"] == "validation" and row["tier"] == "hierarchical"
        ]
        episodes = [int(row["episode"]) for row in validation]
        axes[0].plot(
            episodes,
            [float(row["team_task_reward"]) for row in validation],
            marker="o",
            label=label,
            color=color,
        )
        axes[1].plot(
            episodes,
            [100 * float(row["feasible_deadline_success_rate"])
             for row in validation],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].set_title("Validation reward (closer to 0 is better)", weight="bold")
    axes[1].set_title("Feasible-task deadline success", weight="bold")
    axes[0].set_ylabel("Mean per-task reward")
    axes[1].set_ylabel("Success (%)")
    for axis in axes:
        axis.set_xlabel("Training episode")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Validation curves; V9 validation averages seven topologies",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.root
    figure_dir = root / "docs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    architecture_figure(figure_dir / "rl_architecture.png")
    baseline_figure(
        read_csv(root / "results" / "alibaba-best-tier-calibrated-v8"
                 / "aggregate_metrics.csv"),
        figure_dir / "heldout_policy_comparison_v8.png",
    )
    topology_figures(
        read_csv(root / "results" / "alibaba-topology-sensitivity-v8"
                 / "topology_metrics.csv"),
        read_csv(root / "results" / "alibaba-topology-sensitivity-robust-v9"
                 / "topology_metrics.csv"),
        figure_dir / "topology_robustness.png",
    )
    training_figure(
        read_csv(root / "artifacts" / "models" / "alibaba-best-tier-calibrated-v8"
                 / "training_log.csv"),
        read_csv(root / "artifacts" / "models"
                 / "alibaba-best-tier-calibrated-topology-robust-v9"
                 / "training_log.csv"),
        figure_dir / "validation_curves.png",
    )
    print(f"Generated figures in {figure_dir}")


if __name__ == "__main__":
    main()
