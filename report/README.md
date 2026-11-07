# IEEE article

This directory contains the revised IEEE article for the hierarchical
device--edge--cloud offloading project. It uses the repository's canonical
SimPy backend and incorporates the documented Alibaba-derived DQN/Double-DQN
campaign without modifying simulator or RL source code.

## Compile

From the repository root, with a TeX installation and `latexmk` available:

```bash
python3 report/plot_final_results.py
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The output is `report/main.pdf`. If `latexmk` is unavailable, run `pdflatex`
twice and then `bibtex main`, followed by two more `pdflatex` runs.

## Evidence policy

The numerical RL campaign is documented in
`docs/rl_experiment_report.md`; raw trace files, checkpoints, and campaign
CSV files are excluded from Git. The paper labels those values as documented
experimental findings rather than independently regenerated local artifacts.
The earlier `results/dev-smoke/` CSVs remain a separate preliminary synthetic
baseline and are not statistically combined with the RL campaign.

The architecture figures are explanatory diagrams. The DQN/Double-DQN,
topology, and feedback figures plot only means documented in
`docs/rl_experiment_report.md`; they are not independently regenerated from
local campaign CSV files.

The primary six-policy figures instead read
`docs/experiments/final-matched/summary.csv`. That package contains the
verified aggregate comparison and metadata; raw task traces and checkpoints
are not required by the plotting script.
