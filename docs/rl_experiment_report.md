# Hierarchical task offloading with DQN and Double DQN

## Results available

The evaluation replays 579 chronologically held-out Alibaba tasks on
the same 24-device, three-edge and shared-cloud configuration.
Both algorithms use next-decision credit. DQN takes the maximum value
from the target network. Double DQN selects the next action with the
online network and evaluates it with the target network. The target
differs only for transitions that bootstrap. The two learners use the
same episode cap and validation rule. Early stopping can result in
different numbers of episodes.

All 30 Double DQN conditions are complete. The DQN results below cover
five matched seeds per mode for no, periodic and frequent feedback.
The DQN event campaign is reported separately until each threshold
has five evaluated checkpoints.

| Feedback | DQN success | Double DQN success | DQN P95 | Double DQN P95 | DQN device energy | Double DQN device energy |
|---|---:|---:|---:|---:|---:|---:|
| None | 75.01% | 73.39% | 16.79 s | 16.05 s | 0.211 J | 0.210 J |
| Periodic | 74.28% | 73.49% | 16.90 s | 17.36 s | 0.331 J | 0.211 J |
| Frequent | 74.63% | 74.33% | 16.91 s | 15.08 s | 0.260 J | 0.244 J |

Metrics are means of five training-seed means, each averaging five
noise runs. Paired differences are Double DQN minus DQN. Descriptive
percentile intervals use 2,000 bootstrap resamples of the five paired
training seeds.

| Feedback | Success difference (pp) | 95% interval | P95 difference (s) | 95% interval |
|---|---:|---:|---:|---:|
| None | −1.62 | [−4.00, 1.35] | −0.74 | [−2.98, 1.54] |
| Periodic | −0.79 | [−2.97, 1.66] | +0.46 | [−1.77, 2.36] |
| Frequent | −0.30 | [−1.29, 0.76] | −1.83 | [−6.95, 2.74] |

DQN has higher mean success in these three modes. Double DQN has lower
mean P95 without feedback and with frequent feedback, and uses less
mean device energy in each mode. All paired success and P95 intervals
include zero. They describe variability across the five seeds on one
test window. They do not establish equivalence or a general advantage.

The four held-out topologies change the comparison. Each cell below
reports DQN / Double DQN using the same five pairs of training seeds.

| Topology | Feedback | Success (%) | P95 (s) |
|---|---|---:|---:|
| Faster cloud | None | 87.56 / 86.47 | 7.73 / 8.65 |
| Faster cloud | Periodic | 86.99 / 85.87 | 7.40 / 8.20 |
| Faster cloud | Frequent | 86.56 / 86.78 | 7.92 / 7.91 |
| Faster device–edge link | None | 74.96 / 64.41 | 18.06 / 18.31 |
| Faster device–edge link | Periodic | 68.84 / 68.84 | 19.29 / 20.62 |
| Faster device–edge link | Frequent | 68.61 / 68.28 | 21.52 / 22.80 |
| Faster edge–cloud link | None | 73.24 / 62.62 | 18.13 / 18.45 |
| Faster edge–cloud link | Periodic | 67.27 / 66.93 | 19.50 / 20.75 |
| Faster edge–cloud link | Frequent | 67.14 / 67.03 | 21.73 / 22.83 |
| Combined stress | None | 50.72 / 59.22 | 35.73 / 27.81 |
| Combined stress | Periodic | 61.80 / 57.86 | 37.53 / 37.50 |
| Combined stress | Frequent | 61.73 / 60.17 | 29.42 / 40.41 |

Under combined stress, Double DQN has the higher mean success
without feedback. DQN has the higher mean success with periodic and
frequent feedback. The latency advantage also changes with the mode.
These are scenario-specific descriptive comparisons.

For Double DQN, evaluating each event checkpoint at its own trained
threshold gives 74.29%, 74.95% and 74.98% success for thresholds 0.1,
0.25 and 0.5. The corresponding P95 values are 17.00, 17.46 and 17.80 s.
The event threshold of 0.5 uses 10,430 feedback bytes per run, versus
35,126 with frequent feedback. This is a reduction of about 70% in
feedback bytes, accompanied by a higher P95. Comparisons across modes
include their different feedback costs in the reward.

The event comparison evaluates each checkpoint at its trained
threshold. Each row below averages five training seeds and five
noise runs per seed on the reference topology.

| Threshold | Algorithm | Success | P95 | Device energy per task | Feedback bytes per run | Mean feedback age |
|---|---|---:|---:|---:|---:|---:|
| 0.25 | DQN | 73.07% | 16.61 s | 0.234 J | 11,988 | 65.54 s |
| 0.25 | Double DQN | 74.95% | 17.46 s | 0.271 J | 11,087 | 69.37 s |
| 0.5 | DQN | 74.75% | 16.74 s | 0.329 J | 10,687 | 67.97 s |
| 0.5 | Double DQN | 74.98% | 17.80 s | 0.213 J | 10,431 | 68.80 s |

At threshold 0.25, the paired Double DQN minus DQN difference is
+1.88 percentage points in success (descriptive 95% bootstrap
interval +0.37 to +3.25) and +0.85 s in P95 (−1.80 to +3.48).
At 0.5, the differences are +0.23 points (−1.18 to +1.68) and
+1.05 s (−0.54 to +2.76). Feedback age is averaged over snapshots
consumed at decisions.

## Cloud rule ablation on sequential-credit checkpoints

Use the five frequent-feedback checkpoints for each algorithm. For each
checkpoint, evaluate the learned cloud and EDF, SPT and FIFO fixed rules
with identical device and edge weights, arrivals, deadlines and noise
seeds. The upstream decisions can respond to the queue changes caused
by the cloud scheduler. Pair the resulting trajectories by checkpoint,
topology and noise seed, then average noise within each training seed.
Fixed-rule comparisons are made within each checkpoint. Across the
two learning algorithms, upstream weights can differ even with a
matched training seed.

The replay logs rule selections only when at least two jobs wait in
the cloud queue. Selecting multiple rules within one checkpoint shows
that its decisions vary across the observed queue states. Different rules
can still select the same job, so rule counts alone do not establish a
change in dispatch order or an improvement over a fixed rule.
The cloud learner sees a normalized negative cost based on queue
length multiplied by the selected job's estimated service time. Its
next-decision target links successive contested dispatches. This is
a proxy for queue occupancy during service. Arrivals during service
are not included in the immediate cost.
At a given dispatch, queue length is fixed across candidate actions.
The immediate cost therefore favors shorter jobs. Any benefit from
choosing a different rule would have to appear through subsequent
contested states. The cloud cost also lacks an explicit term for
deadline misses, so the policy's training objective and the
checkpoint selection metric differ.

The replay uses `compare_integrated_policies.py --cloud-rule-ablation`,
with feedback mode `frequent` and evaluation seeds 10001 to 10005.
Outputs are under `results/cloud-rule-sequential/`, one folder per
algorithm and training seed. The learned rule counts are in
`cloud_contested_actions.csv`.

| Frozen upstream weights | Cloud rule | Reference success | Reference P95 | Combined stress success | Combined stress P95 |
|---|---|---:|---:|---:|---:|
| DQN | Learned | 74.63% | 16.91 s | 61.73% | 29.42 s |
| DQN | EDF | 74.65% | 16.37 s | 61.63% | 28.88 s |
| DQN | SPT | 75.23% | 12.26 s | 62.02% | 26.19 s |
| DQN | FIFO | 74.25% | 18.25 s | 61.59% | 30.94 s |
| Double DQN | Learned | 74.33% | 15.08 s | 60.17% | 40.41 s |
| Double DQN | EDF | 74.24% | 15.29 s | 60.17% | 40.25 s |
| Double DQN | SPT | 74.82% | 12.36 s | 60.52% | 39.05 s |
| Double DQN | FIFO | 73.90% | 17.82 s | 60.14% | 41.32 s |

SPT has the lowest mean P95 and the highest mean deadline success in
each of the five tested topologies for both sets of frozen upstream
weights. Across those topologies, every checkpoint selects one rule
whenever multiple jobs wait.

| Algorithm | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|---|
| DQN | FIFO | FIFO | EDF | SPT | EDF |
| Double DQN | SPT | FIFO | EDF | SPT | EDF |

The queue reward and next-decision target therefore still produce a
constant cloud rule per checkpoint under these conditions.
For all ten checkpoints, the replay with the corresponding fixed
rule exactly reproduces every per-task result field, apart from the
policy label, across 14,475 task evaluations per checkpoint.

On the reference topology, the five Double DQN checkpoints achieve
74.33% deadline success and 15.08 s mean P95 with the learned cloud.
With those same upstream weights, SPT fixed achieves 74.82% and
12.36 s, EDF fixed 74.24% and 15.29 s, and FIFO fixed 73.90% and
17.82 s. Across all five topologies, each checkpoint selects one
rule whenever multiple jobs wait. Seeds 42 and 45 select SPT,
43 selects FIFO, and 44 and 46 select EDF. This behavior persists
with the queue reward and next-decision target in the tested runs.

## Experimental protocol

### 1. Conditions and what they test

There are six training conditions per training seed: `none`, `periodic`,
`frequent`, and `event` at thresholds 0.1, 0.25, and 0.5. Five training seeds
(42–46) therefore require 30 runs per algorithm.

| Condition | Trigger | Interpretation |
|---|---|---|
| `none` | No feedback messages | Measures hierarchical control without inter-tier queue snapshots. |
| `periodic` | Initial snapshot, then every five channel observations | Fixed cadence in decision observations; five observations are not a fixed number of seconds. |
| `frequent` | Every observation | High-information/high-control-traffic reference. |
| `event` | Initial snapshot; signal change reaches threshold; or five-observation fallback refresh | Measures the responsiveness/communication trade-off at thresholds 0.1, 0.25, and 0.5. |

The event threshold applies to the change in the queue signal, not to deadline
success or a percentage of tasks. A lower threshold generally permits more
updates; the fallback prevents a signal from remaining stale indefinitely.
Snapshots are causal: a receiving tier cannot use one until modeled
propagation and serialization have elapsed. The simulator counts those bytes,
delay, and information age, but feedback messages do not contend with
data-plane links. The `periodic` and `event` modes both use a fallback interval
of five observations, but only the latter also triggers on signal changes.

### 2. Trace splits, sampling, and arrival compression

The accepted Alibaba workload has 2,891 chronologically ordered tasks:
1,734 for training, 578 for validation, and 579 for test. Task work is derived
from the mean work of valid completed instances. Payload sizes, MEC capacities,
device assignments, and deadlines remain simulator assumptions; this is not
a measured mobile-MEC workload.

Each training episode samples 100 tasks without replacement from the training
split. The sample changes with the episode seed, as do the simulator's seeded
service/bandwidth perturbations. The default samples tasks rather than taking
a contiguous chronological window. Arrival compression scales the gaps
between arrivals; it does not change task work. A factor below one brings
arrivals closer and increases offered load.

If no explicit factor is supplied, it is calibrated using validation to target
offered load 0.75 relative to 12,500 MIPS. The resulting factor is reused on
the test split without adjustment. Test data are not used to calibrate load or
select checkpoints.

### 3. Learners, training budget, and reward

Each agent uses a two-hidden-layer, 64-unit network; replay capacity 8,000;
batch size 32; learning rate `3e-4`; target-network update every 500
optimization steps; and Smooth L1 loss. Device and edge each have two actions,
the cloud has three scheduling-rule actions, and the centralized controller
has three actions. The centralized learner is trained with the same episode
budget and epsilon schedule, but observes global state, so it is a
performance reference rather than an equal-information comparison.

The maximum budget is 1,000 episodes × 100 tasks. Epsilon is episode-based:
it decreases linearly from 0.90 to 0.05 during the first 75% of the budget,
then stays at 0.05. Early stopping may end a run before the floor is reached.
This shared schedule avoids tying exploration to the different numbers of
decisions made by device, edge, and centralized agents.

The campaign trains the cloud agent and uses `next-decision` credit with
discount `gamma=0.9`: a decision can bootstrap from that same agent's next
logged decision. Singleton cloud queues are omitted from replay because EDF,
SPT, and FIFO select the same job there. For contested dispatches, the cloud
uses a queue-waiting-cost reward; this is distinct from the shared per-task
reward used by other tiers. The shared reward weights are 0.35 latency,
0.25 device energy, 0.35 feasible-deadline miss, and 0.05 feedback cost.
Because feedback cost differs by mode, the modes do not optimize exactly the
same reward function; this is a limitation to retain when interpreting the
ablation.

### 4. Topology variants

Training follows a seeded, shuffled, near-balanced schedule over nine
variants: reference; device CPU at 75% and 125%; edge CPU at 50% and 150%;
cloud CPU at 50%; device–edge bandwidth at 10% with propagation doubled;
edge–cloud bandwidth at 10% with propagation doubled; and an alternate
association swapping devices 2 and 3 between edges. The purpose is to expose
the policy to varied capacities and paths during training.

For this revised campaign, validation uses the reference topology only; it
does not select checkpoints by a topology sweep. Four variants are held out
of training and used only for evaluation: cloud CPU at 150%; device–edge
bandwidth doubled with propagation halved; edge–cloud bandwidth doubled with
propagation halved; and combined stress (all CPU capacities at 75%, both link
bandwidths at 50%, propagation doubled). These are simulated sensitivity
cases, not observed deployments or independent trace periods.

### 5. Validation and checkpoint selection

Every 25 episodes, each controller is evaluated greedily (no exploration or
replay updates) on the same 578 validation tasks under five noise seeds.
Hierarchical and centralized checkpoints are selected independently using
raw deadline success first and lower P95 latency as the tie-breaker. The
printed reward is diagnostic; it is not the selection metric. Patience is
eight validation checks without improvement. Training stops when both
controllers have reached that patience, or at the 1,000-episode budget.

### 6. Test evaluation, comparison policies, and metrics

Each checkpoint is evaluated in its own trained feedback mode and threshold,
without exploration. Each of five evaluation seeds (10001–10005) evaluates
all 579 test tasks on the reference topology and four held-out topology
variants. The evaluation seeds vary simulator perturbations, not the
chronological test window.

Comparison policies include the learned hierarchy, centralized DQN,
random, greedy placement with FIFO or EDF cloud scheduling, and a delayed
greedy policy. They help separate feedback effects from placement/scheduling
effects, but not every baseline has the same observation or learning budget.
In particular, centralized DQN has global state.

The reported measures include raw deadline success; feasible-task deadline
success (tasks that could meet the deadline on at least one route with empty
queues); median and P95 latency; device energy; task communication bytes;
feedback bytes; and feedback information age. Raw success remains the
primary all-task measure. P95 is computed per evaluation run, and information
age is accumulated when a decision consumes a delivered snapshot.

### 7. Aggregation and uncertainty

For each training seed, metrics are first averaged over its five evaluation
seeds. The final summary then reports the mean and sample standard deviation
of the five training-seed means. The 95% interval is a percentile bootstrap
over those five independent training-seed means, using 2,000 resamples.
Evaluation seeds therefore do not masquerade as independent training runs.
With five training seeds, intervals remain uncertain and should be treated as
descriptive rather than universal evidence.

### 8. Outputs and completion criteria

Checkpoints and training/decision logs are under
`artifacts/models/feedback-ablation/`; comparison results are under
`results/feedback-ablation/`. Directory names encode mode, threshold,
algorithm, training seed, and credit mode. For example,
`fb-none-threshold-0.25-double-dqn-seed-43-next-decision/aggregate_metrics.csv`
is just one training-seed condition, even if it contains five evaluation
seeds.

Do not pool results or report confidence intervals until all requested
condition/seed result files exist and the protocol/hash checks pass. After
the runs finish, create the consolidated table from the repository root:

The comparison command writes `aggregate_metrics.csv`, which retains the
evaluation event threshold. The compact `energy_success_bytes.csv` omits that
column and must not be used to compare event thresholds. The summarizer reads
`aggregate_metrics.csv` so each trained threshold and tested threshold remain
separate. After the requested runs finish, create the reference-topology table:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_feedback_ablation.py `
  --results-root results\feedback-ablation `
  --feedback-modes none periodic frequent event `
  --event-thresholds 0.1 0.25 0.5 `
  --algorithms double-dqn `
  --seeds 42 43 44 45 46 `
  --credit-mode next-decision |
  Out-File results\feedback-ablation\summary.csv -Encoding utf8
```

The summary reports means, standard deviations, and bootstrap 95% confidence
intervals across training-seed means for raw deadline success, P95 latency,
device energy, communication/feedback bytes, and feedback age. For the four held-out topology scenarios, rerun with
`--scenario alibaba2017_test:cloud_cpu_high`,
`--scenario alibaba2017_test:device_edge_link_fast`,
`--scenario alibaba2017_test:edge_cloud_link_fast`, or
`--scenario alibaba2017_test:combined_stress`. For a comparison between algorithms, use only conditions with the same
feedback mode, threshold and complete set of paired training seeds.
Use `--algorithms dqn double-dqn --matched-event-only` to retain
each event checkpoint's trained threshold when replay grids differ.
The summarizer checks the remaining configuration and requires one
matched event result per evaluation noise seed.
