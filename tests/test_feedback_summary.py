import json

import pytest
from scripts.summarize_feedback_ablation import summarize_files


def test_summarize_files_aggregates_only_requested_scenario(tmp_path):
    results = tmp_path / "energy_success_bytes.csv"
    results.with_name("metadata.json").write_text(
        json.dumps({"configuration_hash": "same-protocol"}),
        encoding="utf-8",
    )
    results.write_text(
        "scenario,policy,deadline_success_rate,p95_latency_s,mean_device_energy_j,"
        "communication_bytes,feedback_bytes,total_bytes,mean_feedback_age_s\n"
        "base,hierarchical,0.5,1.0,0.2,10,2,12,0.1\n"
        "base,hierarchical,0.7,3.0,0.4,20,4,24,0.3\n"
        "stress,hierarchical,1.0,2.0,0.1,30,6,36,0.2\n",
        encoding="utf-8",
    )

    summary = summarize_files(
        [("periodic", 0.25, "double-dqn", 42, results)], "base"
    )

    assert len(summary) == 1
    assert summary[0]["feedback_mode"] == "periodic"
    assert summary[0]["evaluation_rows"] == 2
    assert summary[0]["training_seed_count"] == 1
    assert summary[0]["deadline_success_rate_mean"] == pytest.approx(0.6)
    assert summary[0]["p95_latency_s_mean"] == pytest.approx(2.0)
    assert summary[0]["mean_device_energy_j_mean"] == pytest.approx(0.3)
