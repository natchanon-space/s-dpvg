import pytest
from tensordict import TensorDict
import torch
from torchrl.envs.batched_envs import ParallelEnv, SerialEnv
from torchrl.envs.utils import check_env_specs

from envs.voting_game import SimpleDpvgEnv, SimpleDpvgConfig, get_vectorized_sdpvg


@pytest.fixture
def n_workers():
    return 4


@pytest.fixture
def env_config():
    return SimpleDpvgConfig(
        n_agents=5,
        k=3,
        l=10,
        passive=0,
    )


def test_parallel_specs(n_workers, env_config):
    env = get_vectorized_sdpvg(env_config, n_workers, mode="p")
    check_env_specs(env)


def test_serial_specs(n_workers, env_config):
    env = get_vectorized_sdpvg(env_config, n_workers, mode="s")
    check_env_specs(env)


# TODO: add more testcases on on