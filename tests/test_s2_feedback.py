import pytest

from mec_offloading.coordination.feedback import FeedbackCoordinator
from mec_offloading.domain.models import NetworkLinkState


def test_event_feedback_suppresses_small_changes_and_refreshes():
    link = NetworkLinkState(100, .1)
    coordinator = FeedbackCoordinator("event", frequency=3)
    values = (0.1, 0.2, 0.6, 0.61, 0.62, 0.63)
    received = [
        coordinator.observe(str(i), direction="edge_to_device", value=value,
                            link=link, now_s=float(i))
        for i, value in enumerate(values)
    ]
    assert received == [None, 0.1, 0.1, 0.6, 0.6, 0.6]
    assert [coordinator.cost(str(i)).messages for i in range(len(values))] == [
        1, 0, 1, 0, 0, 1,
    ]


def test_feedback_reception_delayed_and_cached():
    link = NetworkLinkState(100, .1)
    f = FeedbackCoordinator("periodic", frequency=2)
    assert f.observe("t1", direction="edge_to_device", value=.3, link=link,
                     now_s=1.0) is None
    assert f.observe("t2", direction="edge_to_device", value=.6, link=link,
                     now_s=1.05) is None
    assert f.observe("t3", direction="edge_to_device", value=.7, link=link,
                     now_s=1.11) == .3
    # t3 triggers another message; its value cannot be seen immediately.
    assert f.observe("t4", direction="edge_to_device", value=.9, link=link,
                     now_s=1.15) == .3
    assert f.observe("t5", direction="edge_to_device", value=1.0, link=link,
                     now_s=1.22) == .7
    assert (f.cost("t1").messages, f.cost("t2").messages,
            f.cost("t3").messages) == (1, 0, 1)
    assert f.cost("t1").modeled_delay_s > .1
    assert f.cost("t1").bytes_sent == 32


def test_none_mode_sends_nothing_and_each_direction_has_own_clock():
    link = NetworkLinkState(50, .02)
    f = FeedbackCoordinator("none")
    for now in (0.0, 1.0, 2.0):
        assert f.observe("t", direction="cloud_to_edge", value=.1, link=link,
                         now_s=now) is None
    assert f.cost("t").messages == 0
    with pytest.raises(ValueError, match="chronological"):
        f.observe("t", direction="cloud_to_edge", value=.1, link=link, now_s=.5)
    with pytest.raises(ValueError, match="Unknown"):
        f.observe("t", direction="wrong", value=.1, link=link)


def test_feedback_channels_are_isolated_by_node_pair():
    link = NetworkLinkState(50, .02)
    f = FeedbackCoordinator("frequent")
    assert f.observe(
        "t1", direction="edge_to_device", value=.3, link=link,
        now_s=1.0, scope="edge-1->device-1",
    ) is None
    assert f.observe(
        "t2", direction="edge_to_device", value=.8, link=link,
        now_s=1.05, scope="edge-2->device-3",
    ) is None
    assert f.observe(
        "t3", direction="edge_to_device", value=.4, link=link,
        now_s=1.03, scope="edge-1->device-1",
    ) == .3
    assert f.observe(
        "t4", direction="edge_to_device", value=.9, link=link,
        now_s=1.08, scope="edge-2->device-3",
    ) == .8
    with pytest.raises(ValueError, match="chronological"):
        f.observe(
            "t5", direction="edge_to_device", value=.1, link=link,
            now_s=1.02, scope="edge-1->device-1",
        )
