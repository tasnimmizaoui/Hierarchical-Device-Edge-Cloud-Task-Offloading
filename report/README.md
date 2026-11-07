# IEEE article

This directory contains the auditable article draft for the hierarchical
device--edge--cloud offloading project. It uses the repository's canonical
SimPy backend and does not modify the simulator or RL source code.

## Compile

From the repository root, with a TeX installation and `latexmk` available:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The output is `report/main.pdf`. If `latexmk` is unavailable, run `pdflatex`
twice and then `bibtex main`, followed by two more `pdflatex` runs.

## Evidence policy

The committed `results/dev-smoke/` CSV files support the three-seed synthetic
baseline table. No integrated-RL, Alibaba-trace, or final multi-edge result
table is included because the corresponding run artifacts are not present in
the repository. Such claims are marked as open validation work in the paper.

The figures are explanatory architecture diagrams generated for this draft;
they are not plots of unverified experimental data.
