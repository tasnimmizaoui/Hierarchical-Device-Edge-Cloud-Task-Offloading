"""Build the article's vector figures from the verified experimental summaries."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(__file__).resolve().parent
OUT = REPORT / "figures"
DATA = ROOT / "docs" / "experiments" / "final-matched"
POLICIES = {
    "random": ("Random", "#667481"),
    "greedy_fifo": ("Greedy/FIFO", "#00826D"),
    "greedy_edf": ("Greedy/EDF", "#006B57"),
    "centralized_dqn": ("Centralized DQN", "#215DA8"),
    "hierarchical_dqn": ("Hierarchical DQN", "#AD5217"),
    "hierarchical_double_dqn": ("Hierarchical Double DQN", "#744493"),
}
INK = "#192B40"


def save(figure, name):
    figure.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def plot_comparison():
    summary = pd.read_csv(DATA / "summary.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.12), sharey=True, layout="constrained")
    panels = (
        ("deadline_success_rate", "(a) Deadlines met (%)", 100, False),
        ("p95_latency_s", "(b) P95 latency (s)", 1, True),
        ("mean_device_energy_j", "(c) Device energy (J/task)", 1, True),
    )
    for axis, (metric, title, scale, logarithmic) in zip(axes, panels, strict=True):
        rows = summary[summary.metric == metric].set_index("policy")
        for position, (key, (label, color)) in enumerate(POLICIES.items()):
            row = rows.loc[key]
            deviation = row.baseline_noise_seed_sd if row.training_seeds == 0 else row.training_seed_sd
            axis.errorbar(
                row["mean"] * scale, position, xerr=deviation * scale, color=color,
                fmt="o", markersize=4.5, capsize=2.6, linewidth=1.1,
            )
        if logarithmic:
            axis.set_xscale("log")
        axis.set_title(title, loc="left", fontsize=9)
        axis.grid(axis="x", color="#DCE3EB", linewidth=.65, which="major")
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(POLICIES)), [label for label, _ in POLICIES.values()])
    axes[0].invert_yaxis()
    save(figure, "final_matched_comparison.pdf")


def plot_workload():
    workload = pd.read_csv(REPORT / "workload_summary.csv")
    figure, axis = plt.subplots(figsize=(3.5, 1.95), layout="constrained")
    for i, row in workload.iterrows():
        axis.hlines(i, row.work_p50, row.work_p99, color="#B9CADB", linewidth=2)
    for quantile, marker, color in ((50, "o", "#215DA8"), (95, "s", "#00826D"), (99, "D", "#AD5217")):
        axis.scatter(workload[f"work_p{quantile}"], range(len(workload)),
                     marker=marker, color=color, s=27, label=f"P{quantile}", zorder=3)
    axis.set_yticks(range(3), ["Training (1,734)", "Validation (578)", "Test (579)"])
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Derived compute demand (MI, log scale)")
    axis.grid(axis="x", color="#DCE3EB", linewidth=.65)
    axis.legend(loc="upper center", bbox_to_anchor=(.50, 1.29), ncol=3, frameon=False)
    save(figure, "workload_quantiles.pdf")


def plot_tail():
    tasks = pd.read_csv(DATA / "task_results.csv.gz", usecols=["policy", "end_to_end_latency_s"])
    figure, axis = plt.subplots(figsize=(3.5, 2.05), layout="constrained")
    for key in ("greedy_edf", "centralized_dqn", "hierarchical_dqn", "hierarchical_double_dqn"):
        label, color = POLICIES[key]
        latency = np.sort(tasks.loc[tasks.policy == key, "end_to_end_latency_s"])
        exceedance = np.arange(len(latency), 0, -1) / len(latency)
        axis.step(latency, exceedance * 100, where="pre", color=color, label=label, linewidth=1.15)
    axis.axhline(5, color="#7A8793", linewidth=.8, linestyle="--")
    axis.set(xscale="log", yscale="log", ylim=(.25, 100),
             xlabel="End-to-end latency (s, log scale)", ylabel="Tasks above latency (%)")
    axis.grid(color="#DCE3EB", linewidth=.6)
    axis.legend(loc="lower left", fontsize=6.6, frameon=True, framealpha=.96)
    save(figure, "latency_tail.pdf")


def plot_q_network():
    figure, axis = plt.subplots(figsize=(3.5, 1.65), layout="constrained")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    positions = (.12, .37, .63, .88)
    labels = ("Observation\n11 / 10 / 11", "Hidden 1\n64 + ReLU",
              "Hidden 2\n64 + ReLU", "Q scores\n2 / 2 / 3")
    levels = (.37, .55, .73)
    for left, right in zip(positions[:-1], positions[1:], strict=True):
        for first in levels:
            for second in levels:
                axis.plot([left, right], [first, second], color="#B9CADB", linewidth=.65)
    for x, label, color in zip(
        positions, labels, ("#215DA8", "#00826D", "#00826D", "#AD5217"), strict=True
    ):
        axis.scatter([x] * len(levels), levels, color=color, s=45, zorder=3)
        axis.text(x, .17, label, ha="center", va="center", fontsize=7.7)
    axis.text(.5, .93, "Device / edge / cloud", ha="center", fontsize=8)
    save(figure, "q_network.pdf")


def diagram(height):
    figure, axis = plt.subplots(figsize=(3.5, height))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    figure.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
    return figure, axis


def diagram_box(axis, x, y, width, height, label, fill, *, size=8):
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=.012",
        facecolor=fill, edgecolor="#A8B7C7", linewidth=.65,
    ))
    axis.text(x + width / 2, y + height / 2, label,
              ha="center", va="center", fontsize=size)


def arrow(axis, start, end, color=INK, *, dashed=False):
    axis.annotate("", xy=end, xytext=start, arrowprops={
        "arrowstyle": "-|>", "color": color, "lw": 1.05,
        "linestyle": "--" if dashed else "-",
        "shrinkA": 0, "shrinkB": 0,
    })


def plot_architecture():
    figure, axis = diagram(2.05)
    tiers = (
        (.77, "Device agent", "Execute locally", "#E7F0FA"),
        (.45, "Edge agent", "Execute at edge", "#E1F2ED"),
        (.13, "Cloud agent", "Serve next job", "#F5EBE1"),
    )
    for y, tier, action, color in tiers:
        diagram_box(axis, .035, y, .38, .14, tier, color, size=9)
        diagram_box(axis, .61, y, .35, .14, action, "#F5F7FA")
        arrow(axis, (.43, y + .07), (.59, y + .07))
    for top, bottom, label in ((.75, .61, "Offload to assigned edge"),
                               (.43, .29, "Forward to shared cloud")):
        arrow(axis, (.22, top), (.22, bottom))
        axis.text(.27, (top + bottom) / 2, label, va="center", fontsize=7.8)
    axis.text(.785, .065, "EDF / SPT / FIFO", ha="center", fontsize=7.8)
    save(figure, "architecture.pdf")


def plot_lifecycle():
    figure, axis = diagram(1.8)
    routes = (
        (.77, "Local", ("Wait", "Device CPU", "Done")),
        (.49, "Edge", ("Upload", "Wait", "Edge CPU", "Return")),
        (.21, "Cloud", ("Two uploads", "Wait", "Cloud CPU", "Two returns")),
    )
    fills = {
        "Wait": "#F5EBE1", "Done": "#F5F7FA",
        "Device CPU": "#E1F2ED", "Edge CPU": "#E1F2ED", "Cloud CPU": "#E1F2ED",
    }
    for y, label, phases in routes:
        axis.text(.012, y + .068, label, va="center", fontweight="bold", fontsize=8)
        start, stop, gap = .19, .98, .025
        width = (stop - start - gap * (len(phases) - 1)) / len(phases)
        for index, phase in enumerate(phases):
            x = start + index * (width + gap)
            diagram_box(axis, x, y, width, .136, phase,
                        fills.get(phase, "#E7F0FA"), size=7.2)
    arrow(axis, (.19, .091), (.98, .091), "#667481")
    axis.text(.58, .024, "Order of events  ·  durations vary", ha="center", fontsize=7.5)
    save(figure, "task_lifecycle.pdf")


def plot_coordination():
    figure, axis = diagram(1.95)
    for y, label in ((.69, "Sender"), (.37, "Receiver")):
        axis.text(.015, y, label, va="center", fontsize=8, fontweight="bold")
        arrow(axis, (.25, y), (.97, y), "#667481")
    axis.scatter([.34], [.69], color="#215DA8", s=25, zorder=3)
    axis.text(.34, .80, "Sample queue", ha="center", fontsize=8)
    arrow(axis, (.36, .68), (.64, .39), "#215DA8", dashed=True)
    axis.text(.59, .64, "Link delay", color="#215DA8", fontsize=8)
    axis.scatter([.42, .66, .86], [.37] * 3, color=["#AD5217", "#215DA8", "#00826D"],
                 s=[23, 22, 25], zorder=3)
    for x, label in ((.42, "Decision\nuses old state"),
                     (.66, "Update\narrives"),
                     (.86, "Decision\nuses update")):
        axis.text(x, .26, label, ha="center", va="top", fontsize=7.4)
    axis.annotate("", xy=(.86, .055), xytext=(.34, .055),
                  arrowprops={"arrowstyle": "<->", "color": "#667481", "lw": .8})
    axis.text(.60, .009, "Age at use", ha="center", fontsize=7.6)
    save(figure, "coordination.pdf")


def topology_figure():
    figure, axis = plt.subplots(figsize=(3.5, 1.95), layout="constrained")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    for index, (x, capacity) in enumerate(zip((.17, .50, .83), (5000, 6250, 7500)), start=1):
        axis.annotate("", (x, .57), (x, .76), arrowprops={"arrowstyle": "->", "color": "#667481"})
        axis.annotate("", (.50, .24), (x, .41), arrowprops={"arrowstyle": "->", "color": "#667481"})
        axis.text(x, .84, "8 devices", ha="center", va="center", fontsize=8,
                  bbox={"boxstyle": "round,pad=.38", "fc": "#E7F0FA", "ec": "none"})
        axis.text(x, .49, f"Edge {index}\n{capacity:,} MIPS", ha="center", va="center", fontsize=7.5,
                  bbox={"boxstyle": "round,pad=.38", "fc": "#E1F2ED", "ec": "none"})
    axis.text(.50, .13, "Shared cloud · 1 worker\n20,000 MIPS", ha="center", va="center", fontsize=8,
              bbox={"boxstyle": "round,pad=.4", "fc": "#F5EBE1", "ec": "none"})
    save(figure, "topology_24_devices.pdf")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#A8B7C7",
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    plot_comparison()
    plot_workload()
    plot_tail()
    plot_q_network()
    plot_architecture()
    plot_lifecycle()
    plot_coordination()
    topology_figure()
