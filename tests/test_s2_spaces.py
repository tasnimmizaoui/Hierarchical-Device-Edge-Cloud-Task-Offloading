import numpy as np
import pytest

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
from mec_offloading.rl.spaces import SPECS


def test_registry_matches_current_observation_vectors():
    task = Task("t-1", "d-1", 0, 2, 100, .1, 2)
    device = DeviceState("d-1", 1000, 90, 100, 1.5, .5, .001)
    edge = EdgeState(ComputeNodeState("e", Tier.EDGE, 5000),
                     NetworkLinkState(100, .01), NetworkLinkState(1000, .05))
    cloud = CloudState(ComputeNodeState("c", Tier.CLOUD, 20000))
    queue = QueueSummary(1, 1, 1, 1, 1)
    for name, obs in {
        "device": device_observation(task, device, edge, 0, None),
        "edge": edge_observation(task, edge, 0, None),
        "cloud": cloud_observation(queue, cloud),
        "centralized": centralized_observation(task, device, edge, cloud, 0),
    }.items():
        SPECS[name].validate(obs, 0)
    assert SPECS["cloud"].actions == ("EDF", "SPT", "FIFO")
    with pytest.raises(ValueError):
        SPECS["device"].validate(np.zeros(5))
    with pytest.raises(ValueError):
        SPECS["device"].validate(np.zeros(11), 2)


def test_heavy_tailed_workload_observations_remain_finite_and_bounded():
    task = Task("large", "d-1", 0, 1, 2_000_000_000, .1, 5)
    device = DeviceState("d-1", 800, 90, 100, 1.5, .5, .001)
    edge = EdgeState(
        ComputeNodeState("e", Tier.EDGE, 500, queue_wait_s=2_000_000),
        NetworkLinkState(100, .01),
        NetworkLinkState(1000, .05),
    )
    cloud = CloudState(
        ComputeNodeState("c", Tier.CLOUD, 500, queue_wait_s=4_000_000)
    )
    queue = QueueSummary(20_000, -10_000, 100_000, 2_000_000, 50_000)

    observations = (
        device_observation(task, device, edge, 0, None),
        edge_observation(task, edge, 0, None),
        cloud_observation(queue, cloud),
        centralized_observation(task, device, edge, cloud, 0),
    )
    for observation in observations:
        assert np.isfinite(observation).all()
        assert np.max(np.abs(observation)) <= 1


def test_device_observation_includes_local_queue_wait():
    task = Task("queued", "d-1", 0, 1, 100, .1, 5)
    device = DeviceState(
        "d-1", 800, 90, 100, 1.5, .5, .001, queue_wait_s=10.0, queue_length=2
    )
    edge = EdgeState(
        ComputeNodeState("e", Tier.EDGE, 500),
        NetworkLinkState(100, .01),
        NetworkLinkState(1000, .05),
    )

    observation = device_observation(task, device, edge, 0, None)

    assert observation[5] > 0
    assert SPECS["device"].observation_features[5] == "local_queue_delay_log_scaled_s"


def test_device_route_features_compare_local_and_edge_best_case_slack():
    task = Task("route", "d-1", 0, 1, 2000, .1, 5)
    device = DeviceState("d-1", 800, 90, 100, 1.5, .5, .001)
    edge = EdgeState(
        ComputeNodeState("e", Tier.EDGE, 2000, queue_wait_s=0.25),
        NetworkLinkState(100, .01),
        NetworkLinkState(1000, .05),
    )

    observation = device_observation(task, device, edge, 0, None)

    assert observation.shape == (11,)
    assert observation[2] < observation[6]
    assert SPECS["device"].observation_features[6] == "offload_deadline_service_ratio"


def test_cloud_features_normalize_heavy_tail_queue_signals():
    cloud = CloudState(ComputeNodeState("cloud", Tier.CLOUD, 20_000))
    typical = QueueSummary(2, 1, 2, .2, .5, near_deadline_count=1, workload_ratio=2)
    extreme = QueueSummary(
        20_000, -10_000, 100_000, 2_000_000, 50_000,
        near_deadline_count=20_000, workload_ratio=100_000,
    )

    typical_obs = cloud_observation(typical, cloud)
    extreme_obs = cloud_observation(extreme, cloud)

    assert typical_obs.shape == (11,)
    assert np.isfinite(extreme_obs).all()
    assert np.max(np.abs(extreme_obs)) <= 1
    assert extreme_obs[-2] > typical_obs[-2]
    assert extreme_obs[-1] > typical_obs[-1]
