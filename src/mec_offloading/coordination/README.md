# Feedback coordination

`feedback.py` supports `none`, `periodic`, `frequent`, and `event` modes.
Event mode sends an initial sample, then sends another sample when the
normalized queue signal changes by at least 0.25 or after `frequency`
observations. Transmission delay applies to every message, including
refreshes.

Pour reproduire l'ablation événementielle sur Alibaba 2017 depuis la racine
du dépôt, utiliser les trois graines par défaut et la compression commune.

```bash
python scripts/run_feedback_ablation.py --feedback-modes event --arrival-compression 0.1360670304522077
python scripts/summarize_feedback_ablation.py --feedback-modes none periodic frequent event
```

The coordinator sends congestion signals only between adjacent policy levels.
Signals become observable after propagation and serialization delay and are
delivered lazily at the next policy decision. Message counts, bytes, modeled
delay, age, and sample counts are recorded per task.

Transmission does not reserve a data-plane network resource or block task
execution. The model therefore preserves causal signal availability but does
not represent complete communication-resource contention.
The coordinator defines:

- which feedback fields move between device, edge, and cloud;
- who sends each message and when;
- message payload size in bytes;
- simulated communication delay;
- no-feedback, periodic-feedback, and frequent-feedback configurations;
- message count, bytes, and latency overhead.

Feedback ablations must use the same workload, topology, capacities,
checkpoints, and evaluation seeds. Only cadence or event threshold should
vary.

## Boundary

Coordination must not bypass the policy interfaces in
`mec_offloading.domain.ports`. It should provide local feedback to policies,
not expose the complete global simulator state to lower-level agents.
