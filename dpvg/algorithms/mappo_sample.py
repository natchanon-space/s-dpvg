import torch
import torchrl.modules
from tqdm import tqdm
import matplotlib.pyplot as plt

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv, ParallelEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from dpvg.envs.voting_game import SimpleDpvgEnv, SimpleDpvgConfig, get_vectorized_sdpvg

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# local methods import
from dpvg.algorithms.common import episode_reward_ext


# TODO: 1. initially define all configuation here first
# TODO: 2. define whole training line
# TODO: 3. extract envs related out
# TODO: 3. extract algorithm to be a class
# TODO: 4. save and load


# device
device = "cpu"

# sampling
frame_per_batch = 6_000  # Number of frames collected per training iteration
n_iters = 100
total_frames = frame_per_batch * n_iters

# training
n_epochs = 30  # number of optimization steps per training iteration
minibatch_size = 400  # size of the mini-batches in each optimization steps
lr = 3e-4  # learning rate
max_grad_norm = 1.0  # maximum norm for the gradients

# ppo configuration
clip_epsilon = 0.2  # clip value for ppo loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for GAE (generalized advatage estimation)
entropy_eps = 1e-4  # coefficient of the entropy term in the ppo loss
# other configuaration
share_parameters_policy = False  # homogenous policy when it is true!
share_parameters_critic = False  # homogenous critic when it is true!
mappo = False  # IPPO if False, MAPPO if True

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

# env config
env_config = SimpleDpvgConfig(
    n_agents=5,
    k=3,
    l=6,
    max_steps=100
)

# calculate appropriate number of environments
n_agents = env_config.n_agents
n_envs = frame_per_batch // env_config.max_steps


if __name__ == "__main__":
    
    ## prepare env

    # define batched env
    env = get_vectorized_sdpvg(
        env_config=env_config,
        n_workers=n_envs,
        mode="p",
        flatten_obs=True,
    )
    # adding episode reward
    env = episode_reward_ext(env)
    # check and start env
    check_env_specs(env)
    env.reset()

    print(env.full_action_spec)
    print(env.action_spec)


    ## Defining Model Architecture
    policy_net = torch.nn.Sequential(
        torchrl.modules.MultiAgentMLP(
            # lazy init upon first call
            n_agent_inputs=None,  
            # 2 * n_action_per_agent
            # n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
            n_agent_outputs=env.full_action_spec[env.action_key].shape[-1],
            # number of agents
            n_agents=n_agents,
            # the policies are decentralised
            centralized=False,
            # parameter sharing properties
            share_params=share_parameters_policy,
            # device settign
            device=device,
            # model settings: depth of layer
            depth=2,
            # model settings: node per layer
            num_cells=64,
            # model settings: activation function for all node
            activation_class=torch.nn.Tanh  # or ReLU obviously!
        ),
        # NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        # out_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "logits")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        # in_keys=[("agents", "loc"), ("agents", "scale")],
        in_keys=[("agents", "logits")],
        out_keys=[env.action_key],
        distribution_class=torch.distributions.OneHotCategorical,
        return_log_prob=True,
    )  # log prob for PPO loss
    print("==Running Policy==\n", policy(env.reset()))

    
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,  # for a single value function (of a state obviously!)
        n_agents=n_agents,
        centralized=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=64,
        activation_class=torch.nn.Tanh,
    )
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    print("==Running Critic==\n", critic(env.reset()))


    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        frames_per_batch=frame_per_batch,
        total_frames=total_frames,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frame_per_batch,
            device=device,
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,  # We will sample minibatches of this size
    )


    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(  # we have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    )
    GAE = loss_module.value_estimator

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=lr)


    # training loop
    pbar = tqdm(total=n_iters, desc="episode_length_mean = 0")

    episode_length_mean_list = []
    episode_gini_mean_list = []
    episode_entropy_mean_list = []

    for td in collector:
        td.set(
            ("next", "agents", "done"),
            td.get(("next", "done"))
            .unsqueeze(-1)
            .expand(td.get_item_shape(("next", env.reward_key))),
        )
        td.set(
            ("next", "agents", "terminated"),
            td.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(td.get_item_shape(("next", env.reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        with torch.no_grad():
            GAE(
                td,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data

        data_view = td.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        for _ in range(n_epochs):
            for _ in range(frame_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_grad_norm
                )  # Optional

                optimizer.step()
                optimizer.zero_grad()
        
        collector.update_policy_weights_()

        # Logging
        done = td.get(("next", "done")).squeeze(-1)
        # print(done.shape, td.get(("next", "info", "episode_length")).shape)
        # print(td.get(("next", "info", "episode_length"))[:, :, 0, :][done].mean().item())
        # episode_reward_mean = (
        #     # tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        #     td.get(("next", "info", "episode_length"))[done].mean().item()
        # )
        print("ep_gini", td.get(("next", "info", "episode_gini"))[done].mean().item())
        print("ep_entropy", td.get(("next", "info", "episode_entropy"))[done].mean().item())
        print("ep_length", td.get(("next", "info", "episode_length"))[done].mean().item())
        
        episode_length_mean_list.append(
            td.get(("next", "info", "episode_length"))[done].mean().item()
        )
        episode_entropy_mean_list.append(
            td.get(("next", "info", "episode_entropy"))[done].mean().item()
        )
        episode_gini_mean_list.append(
            td.get(("next", "info", "episode_gini"))[done].mean().item()
        )

        pbar.set_description(f"episode_length_mean = {episode_length_mean_list[-1]}", refresh=False)
        pbar.update()

    plt.plot(episode_length_mean_list)
    plt.xlabel("Training iterations")
    plt.ylabel("Length")
    plt.title("Episode length mean")
    plt.show()

    plt.plot(episode_gini_mean_list)
    plt.xlabel("Training iterations")
    plt.ylabel("Gini")
    plt.title("Episode gini mean")
    plt.show()

    plt.plot(episode_entropy_mean_list)
    plt.xlabel("Training iterations")
    plt.ylabel("Entropy")
    plt.title("Episode entropy mean")
    plt.show()

    # properly close
    env.close()
