import json

import pytest
from mec_offloading.experiment_metadata import configuration_hash
from scripts.summarize_feedback_ablation import summarize_files


def test_summarize_files_aggregates_only_requested_scenario(tmp_path):
    results = tmp_path / "aggregate_metrics.csv"
    results.with_name("metadata.json").write_text(
        json.dumps({"configuration_hash": "same-protocol"}),
        encoding="utf-8",
    )
    results.write_text(
        "scenario,policy,deadline_success_rate,p95_latency_s,mean_device_energy_j,"
        "communication_bytes,feedback_bytes,mean_feedback_age_s,event_threshold\n"
        "base,hierarchical,0.5,1.0,0.2,10,2,0.1,0.25\n"
        "base,hierarchical,0.7,3.0,0.4,20,4,0.3,0.25\n"
        "stress,hierarchical,1.0,2.0,0.1,30,6,0.2,0.25\n",
        encoding="utf-8",
    )

    summary = summarize_files(
        [("periodic", 0.25, "double-dqn", 42, results)], "base"
    )

    assert len(summary) == 1
    assert summary[0]["feedback_mode"] == "periodic"
    assert summary[0]["scenario"] == "base"
    assert summary[0]["evaluation_rows"] == 2
    assert summary[0]["training_seed_count"] == 1
    assert summary[0]["deadline_success_rate_mean"] == pytest.approx(0.6)
    assert summary[0]["p95_latency_s_mean"] == pytest.approx(2.0)
    assert summary[0]["mean_device_energy_j_mean"] == pytest.approx(0.3)
    assert summary[0]["total_bytes_mean"] == pytest.approx(18.0)


def test_summarize_files_keeps_event_evaluation_thresholds_separate(tmp_path):
    results = tmp_path / "aggregate_metrics.csv"
    results.with_name("metadata.json").write_text(
        json.dumps({"configuration_hash": "same-protocol"}),
        encoding="utf-8",
    )
    results.write_text(
        "scenario,policy,deadline_success_rate,p95_latency_s,mean_device_energy_j,"
        "communication_bytes,feedback_bytes,mean_feedback_age_s,event_threshold\n"
        "base,hierarchical_event,0.5,1.0,0.2,10,2,0.1,0.1\n"
        "base,hierarchical_event,0.7,3.0,0.4,20,4,0.3,0.5\n",
        encoding="utf-8",
    )

    summary = summarize_files(
        [("event", 0.1, "double-dqn", 42, results)], "base"
    )

    assert len(summary) == 2
    assert {row["evaluation_event_threshold"] for row in summary} == {"0.1", "0.5"}
    assert all(row["training_seed_count"] == 1 for row in summary)


def test_matched_event_summary_accepts_extra_replay_thresholds(tmp_path):
    files = []
    for seed, thresholds in ((42, [0.1]), (43, [0.1, 0.5])):
        folder = tmp_path / str(seed)
        folder.mkdir()
        results = folder / "aggregate_metrics.csv"
        configuration = {"event_thresholds": thresholds, "test_seeds": [10001]}
        results.with_name("metadata.json").write_text(json.dumps({
            "feedback": "event",
            "training_event_threshold": 0.1,
            "test_seeds": [10001],
            "configuration": configuration,
            "configuration_hash": configuration_hash(configuration),
        }), encoding="utf-8")
        results.write_text(
            "scenario,policy,seed,deadline_success_rate,p95_latency_s,mean_device_energy_j,"
            "communication_bytes,feedback_bytes,mean_feedback_age_s,event_threshold\n"
            "base,hierarchical_event,10001,0.5,1.0,0.2,10,2,0.1,0.1\n"
            + ("base,hierarchical_event,10001,0.9,3.0,0.4,20,4,0.3,0.5\n"
               if 0.5 in thresholds else ""),
            encoding="utf-8",
        )
        files.append(("event", 0.1, "dqn", seed, results))

    summary = summarize_files(files, "base", matched_event_only=True)
    assert len(summary) == 1
    assert summary[0]["training_seed_count"] == 2
    assert summary[0]["evaluation_rows"] == 2
    assert summary[0]["deadline_success_rate_mean"] == pytest.approx(0.5)
