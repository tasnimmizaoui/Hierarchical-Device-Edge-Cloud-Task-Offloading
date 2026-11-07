"""Integrate hierarchical/centralized policies with the canonical SimPy backend.

Each completed task supplies delayed terminal credit to only the agents that
actually acted. We deliberately DO NOT bootstrap from a subsequent unrelated
task observation; this is a one-step, contextual decision formulation until
a real stepable multi-agent simulator is jointly implemented.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.actions import CloudSchedulingRule
from mec_offloading.domain.models import CloudState, QueueSummary, Task, TaskResult
from mec_offloading.rl.credit import team_reward, update_dqn_from_episode  # noqa: F401
from mec_offloading.rl.dqn import DQNAgent
from mec_offloading.rl.policies import (
    CentralizedDQNPolicy,
    CentralizedEdgeAdapter,
    CloudDQNPolicy,
    DecisionAgent,
    DecisionLog,
    DeviceDQNPolicy,
    EdgeDQNPolicy,
)
from mec_offloading.rl.spaces import SPECS
from mec_offloading.simulation import SimPyBackend


def make_agents(
    seed: int = 7, *, batch_size: int = 32, gamma: float = 0.94,
    algorithm: str = "double-dqn",
) -> dict[str, DQNAgent]:
    return {
        name: DQNAgent(spec.observation_dim, len(spec.actions),
                       seed=seed + offset, batch_size=batch_size, gamma=gamma,
                       algorithm=algorithm)
        for offset, (name, spec) in enumerate(SPECS.items())
    }


@dataclass(frozen=True)
class FixedCloudPolicy:
    rule: CloudSchedulingRule

    def choose_rule(self, queue: QueueSummary, cloud: CloudState) -> CloudSchedulingRule:
        return self.rule


def run_hierarchical(tasks: list[Task], agents: Mapping[str, DecisionAgent], *,
                     seed: int, feedback_mode: str = "periodic",
                     feedback_frequency: int = 5, event_threshold: float = 0.25,
                     training: bool = False,
                     freeze_cloud: bool = False,
                     fixed_cloud_rule: CloudSchedulingRule | None = None,
                     backend: SimPyBackend | None = None
                     ) -> tuple[list[TaskResult], dict[str, DecisionLog]]:
    coordinator = FeedbackCoordinator(
        feedback_mode, feedback_frequency, event_threshold=event_threshold
    )
    device = DeviceDQNPolicy(agents["device"], coordinator, training=training)
    edge = EdgeDQNPolicy(agents["edge"], coordinator, training=training)
    if freeze_cloud and fixed_cloud_rule is None:
        fixed_cloud_rule = CloudSchedulingRule.EDF
    cloud = (
        FixedCloudPolicy(fixed_cloud_rule)
        if fixed_cloud_rule is not None
        else CloudDQNPolicy(agents["cloud"], training=training)
    )
    results = (backend or SimPyBackend()).run_episode(
        tasks, device, edge, cloud, seed=seed,
    )
    cloud_log = cloud.log if isinstance(cloud, CloudDQNPolicy) else DecisionLog()
    return results, {"device": device.log, "edge": edge.log, "cloud": cloud_log}


def run_centralized(tasks: list[Task], agent: DecisionAgent, *,
                    seed: int, training: bool = False,
                    backend: SimPyBackend | None = None
                    ) -> tuple[list[TaskResult], DecisionLog]:
    controller = CentralizedDQNPolicy(agent, training=training)
    results = (backend or SimPyBackend()).run_episode(
        tasks, controller, CentralizedEdgeAdapter(controller), controller, seed=seed,
    )
    return results, controller.log
