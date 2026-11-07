"""Plot validation reward and exploration against decisions per tier."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts/models/feedback-budget-long"


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 5), layout="constrained")
    for seed in (42, 43, 44):
        log = pd.read_csv(MODELS / f"fb-frequent-seed-{seed}-terminal/training_log.csv")
        validation = log[(log.split == "validation") & (log.tier == "hierarchical")]
        for column, tier in enumerate(("device", "edge")):
            training = log[(log.split == "train") & (log.tier == tier)].set_index("episode")
            decisions = training.decisions.cumsum()
            axes[0, column].plot(
                decisions.loc[validation.episode], validation.team_task_reward,
                label=f"Graine {seed}",
            )
            axes[1, column].plot(decisions, training.epsilon)
            axes[0, column].axvline(decisions.loc[12], color="grey", alpha=.25)
            axes[1, column].axvline(decisions.loc[12], color="grey", alpha=.25)
            axes[0, column].set_title(tier.capitalize())
            axes[1, column].set_xlabel("Décisions cumulées")
            print(seed, tier, "decision_count", decisions.iloc[-1],
                  "epsilon", round(training.epsilon.iloc[-1], 3))
        best = validation.loc[validation.team_task_reward.idxmax()]
        short_best = validation.loc[validation.episode <= 12, "team_task_reward"].max()
        print(seed, "best_episode", best.episode, "short", short_best,
              "long", best.team_task_reward)
    axes[0, 0].set_ylabel("Récompense de validation")
    axes[1, 0].set_ylabel("Probabilité d'exploration")
    axes[0, 0].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    figure.savefig(ROOT / "docs/figures/validation_budget.png", dpi=200)


if __name__ == "__main__":
    main()
