import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from envs.stats import gini_coef
from envs.voting_game import SimpleDpvgConfig


### greedy policy ###

def greedy_actions(obs_td: TensorDict):
    choices = obs_td["choices"][0]
    return (choices[:, 0] < choices[:, 1]).to(torch.int8)


greedy_policy = TensorDictModule(
    module=greedy_actions,
    in_keys=("agents", "observation"),
    out_keys=("agents", "action"),
)


### gini policy ###

def gini_actions(obs_td: TensorDict):
    """original version of gini actions"""
    choices = obs_td["choices"][0]
    state = obs_td["state"][0]

    results = choices.T + state
    
    gini_action_idx = (gini_coef(results[0]) > gini_coef(results[1]))

    return torch.Tensor([gini_action_idx] * 5).unsqueeze(1)


def gini_actions_min_norm(obs_td: TensorDict):
    choices = obs_td["choices"][0]
    state = obs_td["state"][0]

    results = choices.T + state

    # NOTE: for some reason, it is better without min norm!
    results = results - torch.amin(results, dim=1).unsqueeze(1)
    
    gini_action_idx = (gini_coef(results[0]) > gini_coef(results[1]))

    return torch.Tensor([gini_action_idx] * 5).unsqueeze(1)


gini_policy = TensorDictModule(
    module=gini_actions,
    in_keys=("agents", "observation"),
    out_keys=("agents", "action"),
)


gini_policy_min_norm = TensorDictModule(
    module=gini_actions_min_norm,
    in_keys=("agents", "observation"),
    out_keys=("agents", "action"),
)
