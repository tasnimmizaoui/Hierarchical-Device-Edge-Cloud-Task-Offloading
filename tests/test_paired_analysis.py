from scripts.analyze_paired_results import paired_differences


def test_paired_differences_keep_scenarios_separate(tmp_path):
    results = tmp_path / "task_results.csv"
    results.write_text(
        "scenario,policy,seed,task_id,deadline_met\n"
        "base,hierarchical,1,task-1,true\n"
        "base,greedy,1,task-1,false\n"
        "stress,hierarchical,1,task-1,false\n"
        "stress,greedy,1,task-1,true\n",
        encoding="utf-8",
    )

    assert sorted(paired_differences(results, "hierarchical", "greedy")) == [
        -1.0,
        1.0,
    ]
    assert paired_differences(
        results, "hierarchical", "greedy", scenario="base"
    ) == [1.0]
