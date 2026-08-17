import torch
from tqdm import tqdm
import os
import yaml

from tensordict.nn import TensorDictModule
from torchrl.record import CSVLogger

from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage

from torchrl.envs import EnvBase, check_env_specs

from torchrl.modules import QValueModule, SafeSequential, QMixer, MultiAgentMLP
from torchrl.objectives import QMixerLoss, SoftUpdate


from dpvg.envs.voting_game import SimpleDpvgEnv, SimpleDpvgConfig, get_vectorized_sdpvg
from dpvg.algorithms.common import MarlAlgorithm, episode_reward_ext
from configs.common import dict_to_namespace


class Qmix(MarlAlgorithm):

    device: str = "cpu"

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        # self.env.reward_key = ("agents", "episode_reward")

    def build(self):
        # q module
        module = MultiAgentMLP(
            n_agent_inputs=self.env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=self.env.full_action_spec[self.env.action_key].shape[-1],
            n_agents=self.n_agents,
            centralized=False,
            share_params=False,
            device=self.cfg.device,
            depth=self.cfg.arch.depth,
            num_cells=self.cfg.arch.num_cells,
            activation_class=torch.nn.Tanh
        )
        q_module = TensorDictModule(
            module,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action_value")],
        )
        # value module
        v_module = QValueModule(
            action_value_key=("agents", "action_value"),
            out_keys=[
                ("agents", "action"),
                ("agents", "action_value"),
                ("agents", "chosen_action_value"),
            ],
            action_space="one-hot"
        )
        # combine
        self.policy = SafeSequential(q_module, v_module)
        print("running policy:", self.policy(self.env.reset()))
        # mixer
        self.q_mixer = TensorDictModule(
            QMixer(
                state_shape=(self.n_agents, self.env.observation_spec["agents", "observation"].shape[-1],),
                mixing_embed_dim=self.cfg.arch.embed_dim,
                n_agents=self.n_agents,
                device=self.cfg.device,
            ),
            in_keys=[
                ("agents", "chosen_action_value"),
                ("agents", "observation"),  # suppose to be a state
            ],
            out_keys=[
                "chosen_action_value"
            ],
        )
        # print("running mixer:", self.q_mixer(self.policy(self.env.reset())))

    def train(self, logger = None, checkpoint_iter = None):

        # print("action spec", self.env.action_spec)
        # print("action spec unbatched", self.env.action_spec_unbatched)
        # print("policy shape", self.policy(self.env.reset()).get(("agents", "action")).shape)

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

        self.loss_module = QMixerLoss(
            local_value_network=self.policy,
            mixer_network=self.q_mixer,
            action_space="one-hot",
            loss_function="smooth_l1",
            delay_value=True,
        )
        target_updater = SoftUpdate(self.loss_module, eps=self.cfg.optim.loss_eps)

        self.loss_module.set_keys(
            # reward=self.env.reward_key,
            reward="reward",
            action=self.env.action_key,
            done=("done"),
            terminated=("terminated"),
        )

        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.cfg.optim.lr,
        )


        pbar = tqdm(total=self.cfg.train.n_iters, desc="episode_length_mean = 0")

        # collect data
        iter_counter = 0
        for td in self.collector:
            iter_counter += 1

            team_reward = torch.where(
                td.get(("next", "done")),
                -1.0,
                1.0
            )

            td.set(("next", "reward"), team_reward)

            data_view = td.reshape(-1)
            self.replay_buffer.extend(data_view)

            for _ in range(self.cfg.train.n_epochs):
                for _ in range(self.cfg.train.frames_per_batch // self.cfg.train.minibatch_size):
                    subdata = self.replay_buffer.sample()

                    loss_vals = self.loss_module(subdata)
                    loss_value = (loss_vals["loss"])

                    if logger:
                        prefix = "step"
                        logger.log_scalar(
                            f"{prefix}/loss",
                            loss_value.item(),
                        )

                    loss_value.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(),
                        self.cfg.optim.max_grad_norm,
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    target_updater.step()

            self.collector.update_policy_weights_()

            done = td.get(("next", "done")).squeeze(-1)

            # Logging
            if logger:
                # TODO: change this to tensorboard
                prefix = "iter"
                logger.log_scalar(
                    f"{prefix}/done_ep_count",
                    done.sum().item(),
                )
                logger.log_scalar(
                    f"{prefix}/mean_ep_gini",
                    td.get(("next", "info", "episode_gini"))[done].mean().item(),
                )
                logger.log_scalar(
                    f"{prefix}/mean_ep_entropy",
                    td.get(("next", "info", "episode_entropy"))[done].mean().item(),
                )
                logger.log_scalar(
                    f"{prefix}/mean_ep_length",
                    td.get(("next", "info", "episode_length"))[done].mean().item(),
                )
                logger.log_scalar(
                    f"{prefix}/max_ep_length",
                    td.get(("next", "info", "episode_length"))[done].max().item(),
                )
                logger.log_scalar(
                    f"{prefix}/min_ep_length",
                    td.get(("next", "info", "episode_length"))[done].min().item(),
                )

            # pbar.set_description(f"episode_length_mean = {episode_length_mean_list[-1]}", refresh=False)
            pbar.update()

            if checkpoint_iter and logger:
                # TODO: recheck after implementation of save
                if iter_counter % checkpoint_iter == 0:
                    self.save(
                        log_dir=os.path.join(logger.log_dir, logger.exp_name),
                        suffix=f"{iter_counter}"
                    )

        # TODO: recheck after implementation of save
        if logger:
            self.save(
                log_dir=os.path.join(logger.log_dir, logger.exp_name),
                suffix="final",
            )

    def save(self, log_dir, suffix = None):
        if suffix:
            policy_path = os.path.join(log_dir, f"checkpoint_policy_{suffix}.pt2")
            qmixer_path = os.path.join(log_dir, f"checkpoint_qmixer_{suffix}.pt2")
            torch.save(self.policy.state_dict(), policy_path)
            torch.save(self.q_mixer.state_dict(), qmixer_path)

    def load(self, log_dir, suffix = None):
        policy_path = os.path.join(log_dir, f"checkpoint_policy_{suffix}.pt2")
        qmixer_path = os.path.join(log_dir, f"checkpoint_qmixer_{suffix}.pt2")
        self.policy.load_state_dict(torch.load(policy_path))
        self.policy.eval()
        self.q_mixer.load_state_dict(torch.load(qmixer_path))
        self.q_mixer.eval()


# if __name__ == "__main__":
#     # env config
#     env_config = SimpleDpvgConfig(
#         n_agents=5,
#         k=3,
#         l=6,
#         max_steps=128
#     )

#     # calculate appropriate number of environments
#     frame_per_batch = 1024
#     n_envs = frame_per_batch // env_config.max_steps

#     # example logger
#     logger = CSVLogger(
#         exp_name="default_qmix",
#         log_dir="outs",
#     )
    
#     # define batched env
#     env = get_vectorized_sdpvg(
#         env_config=env_config,
#         n_workers=n_envs,
#         mode="p",  # parallel env
#         flatten_obs=True,
#     )
#     # adding episode reward
#     env = episode_reward_ext(env)
#     env.n_agents = env_config.n_agents
#     # check and start env
#     check_env_specs(env)
#     env.reset()

#     print(env.n_agents)
#     print(env.observation_spec)

#     with open("configs/default_qmix.yaml", "r") as f:
#         cfg = dict_to_namespace(yaml.safe_load(f))


#     qmix = Qmix(env, cfg)
#     qmix.train(logger, checkpoint_iter=5)
