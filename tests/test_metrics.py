import pytest

from mec_offloading.domain.metrics import (
    cpu_energy_j,
    deadline_met,
    execution_time_s,
    radio_energy_j,
    transmission_time_s,
)


def test_transmission_time_uses_decimal_mb_and_mbps():
    # 20 MB = 160 Mb; at 100 Mbps serialization takes 1.6 seconds.
    assert transmission_time_s(20.0, 100.0) == pytest.approx(1.6)


def test_transmission_time_includes_propagation_and_queue_wait():
    assert transmission_time_s(1.0, 8.0, 0.02, 0.03) == pytest.approx(1.05)


def test_execution_time_includes_queue_wait():
    assert execution_time_s(500.0, 1000.0, 0.2) == pytest.approx(0.7)


def test_deadline_boundary_is_inclusive():
    assert deadline_met(12.0, 10.0, 2.0) is True
    assert deadline_met(12.001, 10.0, 2.0) is False


def test_energy_formulas_have_joule_units():
    assert radio_energy_j(2.0, 3.0) == pytest.approx(6.0)
    assert cpu_energy_j(0.001, 500.0) == pytest.approx(0.5)


@pytest.mark.parametrize("args", [(1.0, 0.0), (-1.0, 8.0), (float("nan"), 8.0)])
def test_transmission_rejects_invalid_values(args):
    with pytest.raises(ValueError):
        transmission_time_s(*args)
