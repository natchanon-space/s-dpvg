import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from dpvg.envs.stats import gini_coef
from dpvg.envs.voting_game import SimpleDpvgConfig, repeat_batch_dim


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
    """original version of gini actions (the best for GINI)"""

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


## batched versions

def batch_gini_coef(td: TensorDict) -> Tensor:
    """todo: add explaination, ex [2, 5] --> [2]"""
    n = td.shape[-1]

    sum_pair_diff = 0.0
    for i in range(n):
        sum_pair_diff += torch.sum(
            torch.abs(td - td[..., i].unsqueeze(-1)),
            dim=-1
        )
    return sum_pair_diff / (2.0 * n * torch.sum(td, dim=-1))


def batch_gini_actions(obs_td: TensorDict) -> TensorDict:
    # print(obs_td.shape)
    # extract batched and flattened observation
    n_envs, n_agents = obs_td.shape[:2]
    # print(n_envs, n_agents)
    # extract choice
    choice_td = obs_td[..., :n_agents*2].reshape(n_envs, n_agents, n_agents, 2)
    choice_0 = choice_td[:, 0, :, 0].squeeze(-1)
    choice_1 = choice_td[:, 0, :, 1].squeeze(-1)
    # extract state
    state_td = obs_td[..., n_agents*3:].reshape(n_envs, n_agents, n_agents)
    state_td = state_td[:, 0, :]
    # calculate gini actions
    vote_idx = repeat_batch_dim(
        batch_gini_coef(choice_0 + state_td) > batch_gini_coef(choice_1 + state_td),
        n_agents
    ).long().T
    # print("vote_idx", vote_idx.shape)
    # print("vote index", vote_idx.cpu().numpy())
    actions = torch.nn.functional.one_hot(vote_idx, num_classes=2).float()
    # print("action", actions.shape)
    # return action as tensor dict
    return TensorDict(TensorDict(
        {"agents": {
            "action": actions
        }},
        batch_size=(n_envs, n_agents,)
    ), batch_size=(n_envs,))

batch_gini_policy = TensorDictModule(
    module=batch_gini_actions,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "action")],
)

# ---

def batch_greedy_actions(obs_td: TensorDict) -> TensorDict:
    # extract choices
    n_envs, n_agents = obs_td.shape[:2]
    choice_td = obs_td[:, :, :n_agents*2].reshape(n_envs, n_agents, n_agents, 2)
    # remove redundant choices, keeping batch info
    choice_td = choice_td[:, 0, :]
    # print(choice_td.shape, choice_td[..., 0].shape)
    # gini action with batch dims

    # TEMP: action returns
    # actions = (choice_td[..., 0] < choice_td[..., 1]).to(torch.int8).unsqueeze(-1)
    vote_idx = (choice_td[..., 0] < choice_td[..., 1]).long()
    actions = torch.nn.functional.one_hot(vote_idx, num_classes=2).float()
    # print("vote_idx", vote_idx.shape)
    # print(actions.shape)
    return TensorDict(TensorDict(
        {"agents": {
            "action": actions
        }},
        batch_size=(n_envs, n_agents,)
    ), batch_size=(n_envs,))

batch_greedy_policy = TensorDictModule(
    module=batch_greedy_actions,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "action")],
)
