"""Integrate hierarchical/centralized policies with the canonical SimPy backend.

Each completed task supplies delayed terminal credit to only the agents that
actually acted. We deliberately DO NOT bootstrap from a subsequent unrelated
task observation; this is a one-step, contextual decision formulation until
a real stepable multi-agent simulator is jointly implemented.
"""
from __future__ import annotations

from typing import Mapping


from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.models import Task, TaskResult
from mec_offloading.rl.credit import RewardWeights, team_reward, update_dqn_from_episode
from mec_offloading.rl.dqn import DQNAgent
from mec_offloading.rl.spaces import SPECS
from mec_offloading.rl.policies import (
    CentralizedDQNPolicy, CentralizedEdgeAdapter, CloudDQNPolicy,
    DeviceDQNPolicy, EdgeDQNPolicy, DecisionLog,
)
from mec_offloading.simulation import SimPyBackend




def make_agents(seed: int = 7, *, batch_size: int = 32) -> dict[str, DQNAgent]:
    return {
        name: DQNAgent(spec.observation_dim, len(spec.actions),
                       seed=seed + offset, batch_size=batch_size)
        for offset, (name, spec) in enumerate(SPECS.items())
    }


def run_hierarchical(tasks: list[Task], agents: Mapping[str, DQNAgent], *,
                     seed: int, feedback_mode: str = "periodic",
                     feedback_frequency: int = 5, training: bool = False,
                     backend: SimPyBackend | None = None
                     ) -> tuple[list[TaskResult], dict[str, DecisionLog]]:
    coordinator = FeedbackCoordinator(feedback_mode, feedback_frequency)
    device = DeviceDQNPolicy(agents["device"], coordinator, training=training)
    edge = EdgeDQNPolicy(agents["edge"], coordinator, training=training)
    cloud = CloudDQNPolicy(agents["cloud"], training=training)
    results = (backend or SimPyBackend()).run_episode(
        tasks, device, edge, cloud, seed=seed,
    )
    return results, {"device": device.log, "edge": edge.log, "cloud": cloud.log}


def run_centralized(tasks: list[Task], agent: DQNAgent, *,
                    seed: int, training: bool = False,
                    backend: SimPyBackend | None = None
                    ) -> tuple[list[TaskResult], DecisionLog]:
    controller = CentralizedDQNPolicy(agent, training=training)
    results = (backend or SimPyBackend()).run_episode(
        tasks, controller, CentralizedEdgeAdapter(controller), controller, seed=seed,
    )
    return results, controller.log

