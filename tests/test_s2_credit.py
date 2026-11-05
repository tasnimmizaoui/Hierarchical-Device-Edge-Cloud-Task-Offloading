"""Tests that do not need the Student 1 SimPy dependency."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mec_offloading.domain.models import Task, TaskResult, Tier
from mec_offloading.rl.credit import RewardWeights, team_reward, update_dqn_from_episode
from mec_offloading.rl.dqn import DQNAgent, Transition
from mec_offloading.rl.policies import DecisionLog


def outcomes():
    tasks = [Task(f"t-{i}", "device-1", float(i), 1, 100, .1, 2)
             for i in range(3)]
    results = [TaskResult(task.task_id, (Tier.DEVICE,), task.arrival_time_s,
                          task.arrival_time_s + .3, .3, True, .1, 0, 0)
               for task in tasks]
    return tasks, results


class RecordingAgent:
    def __init__(self):
        self.transitions = []

    def observe(self, transition):
        self.transitions.append(transition)


def test_every_decision_receives_its_own_terminal_task_reward():
    tasks, results = outcomes()
    log = DecisionLog()
    log.add("t-2", np.array([2, 3], dtype=np.float32), 0)
    log.add("t-0", np.array([4, 5], dtype=np.float32), 1)
    agent = RecordingAgent()
    reward = update_dqn_from_episode(agent, log, tasks, results)
    assert len(agent.transitions) == 2  # tier not visited by task t-1
    assert all(t.done for t in agent.transitions)
    assert all(np.array_equal(t.next_state, np.zeros(2)) for t in agent.transitions)
    assert agent.transitions[0].state.tolist() == [2, 3]
    assert reward == pytest.approx(team_reward(tasks[0], results[0]))


def test_missing_or_duplicate_log_entries_are_rejected():
    tasks, results = outcomes()
    log = DecisionLog()
    log.add("t-0", np.zeros(2, dtype=np.float32), 0)
    log.add("t-0", np.zeros(2, dtype=np.float32), 0)
    with pytest.raises(ValueError, match="Duplicate"):
        update_dqn_from_episode(RecordingAgent(), log, tasks, results)
    log2 = DecisionLog()
    log2.add("unknown", np.zeros(2, dtype=np.float32), 0)
    with pytest.raises(ValueError, match="Missing"):
        update_dqn_from_episode(RecordingAgent(), log2, tasks, results)


def test_missed_deadline_reduces_reward_and_normalization_checked():
    tasks, results = outcomes()
    good = team_reward(tasks[0], results[0])
    missed = team_reward(tasks[0], replace(results[0], deadline_met=False))
    assert missed == pytest.approx(good - RewardWeights().deadline_miss)
    with pytest.raises(ValueError):
        team_reward(tasks[0], results[0], RewardWeights(energy_reference_j=0))


def test_double_dqn_replay_terminal_shapes_and_loss(tmp_path):
    agent = DQNAgent(3, 2, seed=1, batch_size=2, target_interval=1)
    for action in (0, 1, 0):
        obs = np.asarray([1., 0., .5], dtype=np.float32)
        agent.observe(Transition(obs, action, -1.0, np.zeros_like(obs), True))
    assert agent.learn_steps >= 1
    assert all(t.done for t in agent.memory)
    losses = agent.pop_training_losses()
    assert losses and all(v >= 0 for v in losses)
    assert agent.pop_training_losses() == []
    path = tmp_path / "model.pt"
    agent.save(path)
    loaded = DQNAgent.load(path)
    assert loaded.act(np.zeros(3, dtype=np.float32)) in (0, 1)
    with pytest.raises(ValueError, match="observation shape"):
        agent.observe(Transition(np.zeros(2), 0, 0.0, np.zeros(2), True))
