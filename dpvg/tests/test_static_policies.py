import pytest
import torch

from envs.static_policies import greedy_policy, gini_policy, gini_coef
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


### tests for greedy policy ###

def test_greedy_action_correctness(env):
    rollouts = env.rollout(1000, greedy_policy)

    for rollout in rollouts[1:]:
        choices = rollout["agents"]["observation"]["choices"][0]
        actions = rollout["agents"]["action"]

        print(choices, actions)
        greedy_actions = choices[:, 0] < choices[:, 1]

        assert (greedy_actions == actions).all()


### tests for gini policy ###

def test_gini_action_correctness(env):
    rollouts = env.rollout(1000, gini_policy)

    for rollout in rollouts[1:]:
        choices = rollout["agents"]["observation"]["choices"][0]
        actions = rollout["agents"]["action"]
    
        # check if all votes are unanimous
        assert (actions == actions[0]).all()

        # next step is actually has lower gini coef compared to its alternative
        state = rollout["agents"]["observation"]["state"][0]
        next_state = state + choices.T[int(actions[0])]
        alternative = state + choices.T[1-int(actions[0])]
        # alternative = alternative - torch.min(alternative)
        assert gini_coef(next_state) <= gini_coef(alternative)
