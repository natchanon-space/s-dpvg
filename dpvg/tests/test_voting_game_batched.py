import pytest
from tensordict import TensorDict
import torch
from torchrl.envs.batched_envs import ParallelEnv, SerialEnv
from torchrl.envs.utils import check_env_specs

from envs.voting_game import SimpleDpvgEnv, SimpleDpvgConfig


@pytest.fixture
def n_workers():
    return 4


def create_env_func():
    return SimpleDpvgEnv(
        env_config=SimpleDpvgConfig(
            n_agents=5,
            k=3,
            l=10,
            passive=0,
        )
    )


def test_parallel_specs(n_workers):
    env = ParallelEnv(
        num_workers=n_workers,
        create_env_fn=create_env_func
    )
    check_env_specs(env)


def test_serial_specs(n_workers):
    env = SerialEnv(
        num_workers=n_workers,
        create_env_fn=create_env_func
    )
    check_env_specs(env)
