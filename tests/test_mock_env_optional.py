"""Optional smoke test: runs when the RL extra (Gymnasium) is installed."""

import pytest


def test_device_mock_env_obeys_gymnasium_api():
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from mec_offloading.rl.envs.device_mock_env import DeviceOffloadingMockEnv

    env = DeviceOffloadingMockEnv()
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info["mock_environment"] is True

    _, reward, terminated, truncated, step_info = env.step(0)
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert step_info["is_mock"] is True
    check_env(env, skip_render_check=True)
