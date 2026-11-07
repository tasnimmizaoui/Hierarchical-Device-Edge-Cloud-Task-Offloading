"""Generate trace exploration figures and summary with the production adapter."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from mec_offloading.rl.credit import deadline_feasibility
from mec_offloading.rl.experiment_data import load_trace_experiment
from mec_offloading.simulation import SimPyBackend

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures"


def main() -> None:
    data = load_trace_experiment(
        ROOT / "data/raw/batch_task.csv",
        ROOT / "data/raw/batch_instance.csv",
        ROOT / "configs/alibaba-multi-topology.yaml",
        max_tasks=5000,
        deadline_policy="best-tier-calibrated",
    )
    figure, axes = plt.subplots(2, 2, figsize=(9, 5.8), layout="constrained")
    backend = SimPyBackend(config=data.simulation_config)
    rows = []
    for name, tasks in [
        ("Entraînement", data.train),
        ("Validation", data.validation),
        ("Test", data.test),
    ]:
        times = np.array([task.arrival_time_s for task in tasks])
        work = np.array([task.required_mi for task in tasks])
        payload = np.array([task.input_size_mb for task in tasks])
        deadline = np.array([task.deadline_s for task in tasks])
        gaps = np.diff(times)
        feasible = deadline_feasibility(tasks, backend)
        row = {
            "split": name,
            "n": len(tasks),
            "span_s": times[-1] - times[0],
            "same_timestamp_fraction": np.mean(gaps == 0),
            "deadline_floor_fraction": np.mean(deadline == 1),
            "nominal_feasible_fraction": np.mean(list(feasible.values())),
            "top_1_percent_work_share": sum(sorted(work)[-int(np.ceil(len(work) / 100)):]) / sum(work),
        }
        for ax, values, label in zip(
            axes.flat,
            (work, payload, gaps, deadline),
            ("Calcul (MI)", "Entrée (Mo)", "Interarrivée brute (s)", "Deadline (s)"),
            strict=True,
        ):
            ordered = np.sort(values)
            ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                    where="post", label=name)
            ax.set_xlabel(label)
            ax.set_ylabel("Fraction cumulée")
            if label.startswith("Inter"):
                ax.set_xscale("symlog", linthresh=1)
            else:
                ax.set_xscale("log")
            ax.grid(alpha=.25)
        for field, values in [("work", work), ("payload", payload), ("gap", gaps), ("deadline", deadline)]:
            for percentile in (50, 95, 99):
                row[f"{field}_p{percentile}"] = np.percentile(values, percentile)
        rows.append(row)
    axes[0, 0].legend(fontsize=8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "trace_eda.png", dpi=200)
    table = pd.DataFrame(rows)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    table.to_csv(results / "report_trace_eda.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
