# Final matched-policy comparison on Alibaba 2017

This package compares six policies on the reference topology with 24 devices,
three edge nodes and one shared cloud worker. Each replay uses the same 579
held-out tasks, arrival timestamps, resource configuration and deadline
formula. Evaluation seeds 10001 through 10005 vary simulated bandwidth and
service time noise on that single test window.

## Checkpoints and matching

The local inventory contains five complete checkpoint bundles per algorithm
and feedback condition, for 60 bundles across DQN and Double DQN. The
conditions are no feedback, periodic feedback, frequent feedback and event
feedback at thresholds 0.1, 0.25 and 0.5. Each bundle contains device, edge,
cloud and centralized checkpoints. This comparison uses the ten **frequent
feedback** bundles with training seeds 42 through 46, so both hierarchical
learners use the same feedback cadence. Their cloud schedulers are learned.
The centralized reference uses the five **standard DQN** checkpoints from
those same bundles. It observes global state and uses the centralized
controller's fixed EDF at the cloud.

The packager verifies configuration hashes, matching train and replay
parameters apart from the algorithm, trace and topology fingerprints,
observation and reward versions, and disjoint training, validation and
evaluation seed sets. The configuration records the exact version identifiers.
Arrival compression is 0.12793575549650835, calibrated on validation.
The deadline formula is the best-tier calibrated rule, with the original
one-second floor. Task IDs and arrival timestamps are checked across every
exported replay.

## Pilot and full replay

In the Alibaba branch of the comparison script, the trace loader reads
the YAML passed to the topology option. For each scenario and noise seed,
the script prepares the tasks once and gives every policy a fresh SimPy
backend built from the same configuration. The nominal scenario uses the
reference YAML directly. The synthetic scenario path is separate.

Both fresh pilots completed at seed 10001 using checkpoint seed 42 and
20 matched tasks per policy on the reference configuration. The comparator
also ran its four topology variants. This package filters to the reference
and the six requested policies. The pilot additionally exercises delayed
Greedy, which is emitted by the comparator.

The full replays already existed. The commands below reproduce the
reference rows in their source bundles. The script evaluates the variants
as well, and the packager retains only the reference rows. Run from the
repository root with the project virtual environment, the local trace
CSVs and the trained checkpoint bundles.

```powershell
foreach ($algorithm in 'dqn','double-dqn') {
    $model = "artifacts/models/feedback-ablation/fb-frequent-threshold-0.25-$algorithm-seed-42-next-decision"
    .venv/Scripts/python.exe scripts/compare_integrated_policies.py --source alibaba2017 --topology-config configs/alibaba-multi-topology.yaml --deadline-policy best-tier-calibrated --feedback frequent --tasks 20 --max-trace-tasks 5000 --seeds 10001 --models $model --output "results/final-matched-pilot/$algorithm"
}
```

```powershell
foreach ($algorithm in 'dqn','double-dqn') {
    foreach ($seed in 42..46) {
        $run = "fb-frequent-threshold-0.25-$algorithm-seed-$seed-next-decision"
        .venv/Scripts/python.exe scripts/compare_integrated_policies.py --source alibaba2017 --topology-config configs/alibaba-multi-topology.yaml --deadline-policy best-tier-calibrated --feedback frequent --feedback-modes frequent --feedback-frequency 5 --event-thresholds 0.1 0.25 0.5 --tasks 5000 --max-trace-tasks 5000 --seeds 10001 10002 10003 10004 10005 --models "artifacts/models/feedback-ablation/$run" --output "results/final-matched-replay/$run"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
```

The task argument is a cap. The test split contains 579 tasks, so the
full command uses all 579. This is the packaging command used for this
PR. To package a fresh full replay, add
`--results-root results/final-matched-replay`.

```powershell
.venv/Scripts/python.exe scripts/package_matched_comparison.py --output docs/experiments/final-matched
```

## Results

The baseline rows are deduplicated after checking they agree across all
source bundles. Their standard deviation measures variation over five
evaluation noise seeds. For each learned policy, the five noise runs are
averaged within each training seed before computing the mean and sample
standard deviation across the five training seeds. P95 is computed per
replay before averaging. The two kinds of standard deviation describe
different sources of variation.

| Policy | Deadline success mean ± SD | P95 mean ± SD | Device J/task mean ± SD |
|---|---:|---:|---:|
| Random | 49.57 ± 0.95% | 137.96 ± 16.46 s | 10.480 ± 0.720 |
| Greedy/FIFO | 75.79 ± 0.78% | 13.62 ± 0.49 s | 0.211 ± 0.001 |
| Greedy/EDF | 75.89 ± 0.86% | 12.95 ± 0.43 s | 0.211 ± 0.001 |
| Centralized DQN | 77.41 ± 1.37% | 15.71 ± 4.44 s | 0.242 ± 0.067 |
| Hierarchical DQN | 74.63 ± 0.79% | 16.91 ± 3.32 s | 0.260 ± 0.104 |
| Hierarchical Double DQN | 74.33 ± 1.66% | 15.08 ± 2.85 s | 0.244 ± 0.065 |

The deadline success SD is expressed in percentage points. Centralized DQN
has the highest mean success in this comparison. Greedy/EDF has the lowest
mean P95 and shares the lowest mean device energy with Greedy/FIFO. Both
hierarchical learners outperform Random on these three metrics. They have
lower mean success and higher mean P95 and device energy than Greedy/EDF.
Double DQN has a lower mean P95 than hierarchical DQN with a small decrease
in mean success. These descriptive results use one chronological test
window. The centralized and Greedy references have different information
access from the hierarchy, and centralized state collection traffic is
outside the simulator's accounting.

## Files

- [Task results (gzip CSV)](task_results.csv.gz) contains 52,110 rows.
- [Aggregate metrics](aggregate_metrics.csv) contains 90 policy replays.
- [Means and standard deviations](summary.csv) provides nine metrics.
- [Configuration and checkpoint inventory](metadata.json) records the ten
  selected bundles, their hashes and their checkpoint selection metadata.
