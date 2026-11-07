"""Adapter and coordination unit tests, independent of the SimPy package."""
from __future__ import annotations

import numpy as np

from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.actions import CloudSchedulingRule, DeviceAction, EdgeAction
from mec_offloading.domain.models import (
    CloudState,
    ComputeNodeState,
    DeviceState,
    EdgeState,
    NetworkLinkState,
    QueueSummary,
    Task,
    Tier,
)
from mec_offloading.rl.observations import (
    centralized_observation,
    cloud_observation,
    device_observation,
    edge_observation,
)
from mec_offloading.rl.policies import (
    CentralizedDQNPolicy,
    CentralizedEdgeAdapter,
    CloudDQNPolicy,
    DeviceDQNPolicy,
    EdgeDQNPolicy,
)


class FixedAgent:
    def __init__(self, action: int):
        self.action = action
        self.observations: list[np.ndarray] = []

    def act(self, obs: np.ndarray, training: bool = False) -> int:
        assert obs.dtype == np.float32
        self.observations.append(obs)
        return self.action


def fixture():
    t = Task("test-1", "dev-1", 1.0, 2.0, 900.0, 0.1, 3.0)
    d = DeviceState("dev-1", 1000, 80, 100, 1.5, 0.5, 0.001)
    l1 = NetworkLinkState(100, .01)
    l2 = NetworkLinkState(1000, .04)
    e = EdgeState(ComputeNodeState("edge-1", Tier.EDGE, 5000, .4, .5, 3), l1, l2)
    c = CloudState(ComputeNodeState("cloud-1", Tier.CLOUD, 20000, .6, .2, 2))
    q = QueueSummary(2, .5, 2.0, .2, .25)
    return t, d, e, c, q


def test_observation_shapes_and_no_global_signal_leak():
    t, d, e, c, q = fixture()
    dev = device_observation(t, d, e, 1.0, None)
    edg = edge_observation(t, e, 1.0, None)
    assert dev.shape == (11,) and edg.shape == (10,)
    assert dev[-2:].tolist() == [-1.0, 0.0]
    assert edg[-2:].tolist() == [-1.0, 0.0]
    assert cloud_observation(q, c).shape == (11,)
    assert centralized_observation(t, d, e, c, 1.0).shape == (16,)


def test_feedback_counts_only_sent_adjacent_messages():
    t, d, e, c, q = fixture()
    fb = FeedbackCoordinator("periodic", frequency=2)
    assert fb.observe(t.task_id, direction="edge_to_device", value=.4,
                      link=e.device_edge_link, now_s=0.0) is None
    # A packet must not be observable before propagation+serialization completes.
    assert fb.observe("test-2", direction="edge_to_device", value=.9,
                      link=e.device_edge_link, now_s=0.005) is None
    assert fb.observe("test-3", direction="edge_to_device", value=.1,
                      link=e.device_edge_link, now_s=0.02) == .4
    assert fb.cost(t.task_id).messages == 1
    assert fb.cost(t.task_id).bytes_sent == 32
    assert fb.cost("test-2").messages == 0
    assert fb.cost("test-3").messages == 1
    assert fb.cost("test-3").information_age_samples == 1
    assert fb.cost("test-3").information_age_s == .02
    assert FeedbackCoordinator("none").observe(
        t.task_id, direction="cloud_to_edge", value=.3,
        link=e.edge_cloud_link) is None


def test_event_threshold_is_configurable():
    _, _, edge, _, _ = fixture()
    feedback = FeedbackCoordinator("event", frequency=3, event_threshold=0.1)
    feedback.observe("a", direction="edge_to_device", value=0.1,
                     link=edge.device_edge_link, now_s=0.0)
    feedback.observe("b", direction="edge_to_device", value=0.15,
                     link=edge.device_edge_link, now_s=0.01)
    assert feedback.cost("b").messages == 0
    feedback.observe("c", direction="edge_to_device", value=0.25,
                     link=edge.device_edge_link, now_s=0.02)
    assert feedback.cost("c").messages == 1


def test_three_level_agent_adapters_and_cloud_action_mapping():
    t, d, e, c, q = fixture()
    fb = FeedbackCoordinator("frequent")
    device = DeviceDQNPolicy(FixedAgent(1), fb)
    edge = EdgeDQNPolicy(FixedAgent(1), fb)
    cloud = CloudDQNPolicy(FixedAgent(2))
    device.set_time(1)
    edge.set_time(2)
    assert device.choose(t, d, e) == DeviceAction.OFFLOAD_TO_EDGE
    assert edge.choose(t, e, c) == EdgeAction.FORWARD_TO_CLOUD
    assert cloud.choose_rule(q, c) == CloudSchedulingRule.FIFO
    cloud.record_dispatch(t.task_id)
    assert len(cloud.log.entries) == 1
    assert cloud.log.contested_actions == {CloudSchedulingRule.FIFO: 1}
    single_task_queue = QueueSummary(1, .5, .5, .2, .25)
    cloud.choose_rule(single_task_queue, c)
    assert cloud.log.contested_actions == {CloudSchedulingRule.FIFO: 1}
    cloud.record_dispatch("single")
    assert len(cloud.log.entries) == 1
    assert fb.cost(t.task_id).messages == 2


def test_centralized_controller_routes_without_direct_device_cloud():
    t, d, e, c, q = fixture()
    ctrl = CentralizedDQNPolicy(FixedAgent(2))
    assert ctrl.choose_global(t, d, e, c) == DeviceAction.OFFLOAD_TO_EDGE
    assert CentralizedEdgeAdapter(ctrl).choose(t, e, c) == EdgeAction.FORWARD_TO_CLOUD
    assert ctrl.choose_rule(q, c) == CloudSchedulingRule.EDF
    assert len(ctrl.log.entries) == 1
