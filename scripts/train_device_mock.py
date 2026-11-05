"""Train a tiny DQN on the toy environment (RL API smoke test only)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/device_mock_dqn"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit("Install RL dependencies: pip install -e '.[dev,rl]'") from exc

    from mec_offloading.rl.envs.device_mock_env import DeviceOffloadingMockEnv

    env = DeviceOffloadingMockEnv()
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=100,
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        gamma=0.95,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    metadata = {
        "environment": "DeviceOffloadingMockEnv",
        "is_final_model": False,
        "algorithm": "DQN",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "observation_order": [
            "normalized_cpu_demand",
            "normalized_input_size",
            "normalized_relative_deadline",
            "battery_fraction",
            "normalized_local_cpu",
            "normalized_device_edge_bandwidth",
        ],
        "actions": {"0": "EXECUTE_LOCAL", "1": "OFFLOAD_TO_EDGE"},
        "created_at_utc": datetime.now(UTC).isoformat(),
        "note": "Toy environment only; these are not project results.",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved mock model to: {args.output}.zip")


if __name__ == "__main__":
    main()
