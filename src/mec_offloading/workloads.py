"""Seeded synthetic workload generation.

Synthetic values are explicit assumptions; they are not Alibaba trace measurements.
"""

from __future__ import annotations

import random

from mec_offloading.domain.models import Task


def generate_synthetic_tasks(
    count: int = 6,
    *,
    seed: int = 7,
    device_id: str = "device-1",
) -> list[Task]:
    rng = random.Random(seed)
    return [
        Task(
            task_id=f"task-{index + 1}",
            device_id=device_id,
            arrival_time_s=float(index) * 0.05,
            input_size_mb=round(rng.uniform(0.5, 2.0), 3),
            required_mi=round(rng.uniform(250.0, 1_000.0), 3),
            result_size_mb=round(rng.uniform(0.05, 0.2), 3),
            deadline_s=round(rng.uniform(1.0, 4.0), 3),
        )
        for index in range(count)
    ]
