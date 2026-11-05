import numpy as np
import pytest

from mec_offloading.rl.observations import (
    centralized_observation, cloud_observation, device_observation, edge_observation,
)
from mec_offloading.rl.spaces import SPECS
from mec_offloading.domain.models import (
    Task, DeviceState, ComputeNodeState, EdgeState, CloudState,
    NetworkLinkState, QueueSummary, Tier,
)


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
        SPECS["device"].validate(np.zeros(10), 2)
