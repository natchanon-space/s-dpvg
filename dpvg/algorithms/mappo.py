import torch
import torchrl.modules
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector, Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv, ParallelEnv, EnvBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from dpvg.envs.voting_game import SimpleDpvgEnv, SimpleDpvgConfig, get_vectorized_sdpvg

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# local methods import
from dpvg.algorithms.common import MarlAlgorithm, episode_reward_ext
from configs.common import dict_to_namespace

class Mappo(MarlAlgorithm):
    
    device: str = "cpu"
    
    def __init__(self, env: EnvBase, cfg):
        super().__init__()
        set_composite_lp_aggregate(False).set()
        # TODO: refactoring parameters out to be external setting
        # env registration
        self.env = env
        self.n_agents = self.env.n_agents
        self.env.reset()
        self.cfg = cfg

        # build
        self.build()
    
    def build(self):
        ## policy network
        policy_net = torch.nn.Sequential(
            torchrl.modules.MultiAgentMLP(
                # lazy init upon first call
                n_agent_inputs=self.env.observation_spec["agents", "observation"].shape[-1],  
                # 2 * n_action_per_agent
                # n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
                n_agent_outputs=self.env.full_action_spec[self.env.action_key].shape[-1],
                # number of agents
                n_agents=self.n_agents,
                # the policies are decentralised
                centralized=False,  # independent
                # parameter sharing properties
                share_params=self.cfg.ppo.share_parameters_policy,
                # device settign
                device=self.cfg.device,
                # model settings: depth of layer
                depth=self.cfg.arch.depth,
                # model settings: node per layer
                num_cells=self.cfg.arch.num_cells,
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
        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec_unbatched,
            # in_keys=[("agents", "loc"), ("agents", "scale")],
            in_keys=[("agents", "logits")],
            out_keys=[self.env.action_key],  # (agents, actions)
            distribution_class=torch.distributions.OneHotCategorical,
            return_log_prob=True,
        )  # log prob for PPO loss
        self.policy(self.env.reset())
        # print("==Running Policy==\n", self.policy(self.env.reset()))

        ## critic network
        critic_net = MultiAgentMLP(
            n_agent_inputs=self.env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,  # for a single value function (of a state obviously!)
            n_agents=self.n_agents,
            centralized=self.cfg.ppo.centralized_critic,
            share_params=self.cfg.ppo.share_parameters_critic,
            device=self.cfg.device,
            depth=self.cfg.arch.depth,
            num_cells=self.cfg.arch.num_cells,
            activation_class=torch.nn.Tanh,
        )
        self.critic = TensorDictModule(
            module=critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )
        self.critic(self.env.reset())
        # print("==Running Critic==\n", self.critic(self.env.reset()))


    def train(self):
        ## defining collector and buffer
        ## colelctor
        self.collector = Collector(
            self.env,
            self.policy,
            device=self.cfg.device,
            frames_per_batch=self.cfg.train.frames_per_batch,
            total_frames=self.cfg.train.total_frames,
        )
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                self.cfg.train.frames_per_batch,
                device=self.cfg.device,
            ),  # We store the frames_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=self.cfg.train.minibatch_size,
        )

        ## loss module
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=self.cfg.ppo.clip_epsilon,
            entropy_coeff=self.cfg.ppo.entropy_eps,
            normalize_advantage=False  # Important to avoid normalizing across the agent dimension
        )
        self.loss_module.set_keys(
            reward=self.env.reward_key,
            action=self.env.action_key,
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.cfg.ppo.gamma, lmbda=self.cfg.ppo.lmbda
        )
        self.GAE = self.loss_module.value_estimator  # LossModule

        ## optimizer
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=self.cfg.optim.lr)

        ## setting progress bar and stat his
        pbar = tqdm(total=self.cfg.train.n_iters, desc="episode_length_mean = 0")

        episode_length_mean_list = []
        episode_gini_mean_list = []
        episode_entropy_mean_list = []

        ## training loop

        # collect data
        for td in self.collector:  # equivalent to `for each batch`
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

            # compute advantage
            with torch.no_grad():
                self.GAE(
                    td,
                    params=self.loss_module.critic_network_params,
                    target_params=self.loss_module.target_critic_network_params,
                )  # Compute GAE and add it to the data
                # adding "advantage" within td.keys()

            data_view = td.reshape(-1)  # Flatten the batch size to shuffle data
            self.replay_buffer.extend(data_view)  # add to buffer

            # compute
            for _ in range(self.cfg.train.n_epochs):
                for _ in range(self.cfg.train.frames_per_batch // self.cfg.train.minibatch_size):
                    subdata = self.replay_buffer.sample()
                    loss_vals = self.loss_module(subdata)

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # model optimization by step
                    loss_value.backward()  # compute gradient of current tensor

                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.cfg.optim.max_grad_norm
                    )  # Optional

                    self.optimizer.step()  # single opimization step
                    self.optimizer.zero_grad()
            
            self.collector.update_policy_weights_()

            # Logging
            done = td.get(("next", "done")).squeeze(-1)

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
        self.env.close()

    def save(self):
        pass

    def load(self):
        pass

    def close(self):
        """close everything"""
        self.env.close()


if __name__ == "__main__":
    # env config
    env_config = SimpleDpvgConfig(
        n_agents=5,
        k=3,
        l=6,
        max_steps=100
    )

    # calculate appropriate number of environments
    frame_per_batch = 1000
    n_envs = frame_per_batch // env_config.max_steps
    
    # define batched env
    env = get_vectorized_sdpvg(
        env_config=env_config,
        n_workers=n_envs,
        mode="p",  # parallel env
        flatten_obs=True,
    )
    # adding episode reward
    env = episode_reward_ext(env)
    env.n_agents = env_config.n_agents
    # check and start env
    check_env_specs(env)
    env.reset()

    print(env.n_agents)

    with open("configs/default_mappo.yaml", "r") as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    # print(env.full_action_spec)
    # print(env.action_spec)
    mappo = Mappo(env, cfg)
    mappo.train()

    # for param_tensor in mappo.policy.state_dict():
    #     print(param_tensor, "\t", mappo.policy.state_dict()[param_tensor].size)

    mappo.close()
