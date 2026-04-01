import pytest
from tensordict import TensorDict
import torch
from torchrl.envs.utils import check_env_specs

from envs.voting_game import SimpleDpvgEnv, SimpleDpvgConfig


env_config = SimpleDpvgConfig(
    n_agents=5,
    k=3,
    l=10,
    passive=0,
)

### fixtures for tests ###

@pytest.fixture
def env() -> SimpleDpvgEnv:
    return SimpleDpvgEnv(env_config=env_config)

### tests for configuration ###

def test_configuration(env):
    assert env.n_agents == env_config.n_agents
    assert env.k == env_config.k
    assert env.l == env_config.l
    assert env.passive == env_config.passive

def test_check_env_specs(env):
    check_env_specs(env)

### tests for game logic
def _check_single_rollout(td: TensorDict) -> bool:
    # obs + action -> next_obs
    obs = td
    next_obs = td["next"]
    action = obs["agents"]["action"]
    choices = obs["agents"]["observation"]["choices"][0]
    # count votes
    count_1 = torch.sum(action)
    count_0 = action.shape[0] - count_1
    winning_idx = int(count_1 > count_0)
    selected_choice = choices[:, winning_idx]
    # check correctness of generated choices
    assert (choices[:, 0] != choices[:, 1]).all()
    assert torch.logical_and(0 <= choices, choices <= env_config.k).all()
    # check correctness of transition
    next_state = obs["agents"]["observation"]["state"][0] + selected_choice
    next_state = next_state - torch.min(next_state)
    assert (next_state == next_obs["agents"]["observation"]["state"][0]).all()
    # check correctness of reward
    reward = next_obs["agents"]["reward"]
    assert (reward == next_obs["agents"]["reward"]).all()

def _check_rollout_obs_alignment(td1: TensorDict, td2: TensorDict):
    """Check alignment of next obs of td1 and obs of td2."""
    assert (
        td1["next"]["agents"]["observation"] == td2["agents"]["observation"]
    ).all()

def _validate_terminal_condition(td: TensorDict):
    # check transition of termination flag
    assert td["done"] == False
    assert td["next"]["done"] == True
    assert td["terminated"] == False
    assert td["next"]["terminated"] == True
    # check validation of state at terminal state
    last_state = td["next"]["agents"]["observation"]["state"][0]
    assert torch.max(last_state) - torch.min(last_state) > env_config.l


def test_single_rollout(env):
    for _ in range(100):
        rollout = env.rollout(1)[0]
        _check_single_rollout(rollout)

def test_rollout_observation_alignment(env):
    for _ in range(100):
        td1, td2 = env.rollout(2)
        _check_rollout_obs_alignment(td1, td2)

def test_multiple_rollouts(env):
    # generate rollouts
    rollouts = env.rollout(10000)
    rollout_length = len(rollouts)
    assert rollout_length <= 10000
    # validate rollout pairs
    for i in range(rollout_length - 1):
        td1 = rollouts[i]
        td2 = rollouts[i+1]
        _check_single_rollout(td1)
        _check_rollout_obs_alignment(td1, td2)
    # check last rollout and its terminal state
    _check_single_rollout(rollouts[-1])
    _validate_terminal_condition(rollouts[-1])
