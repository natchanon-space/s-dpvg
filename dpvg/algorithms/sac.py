import torch
import torchrl.modules
from tqdm import tqdm
import os


from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from torchrl.record import CSVLogger
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import DiscreteSACLoss, SoftUpdate

from dpvg.algorithms.common import MarlAlgorithm, episode_reward_ext
from configs.common import dict_to_namespace


class SAC(MarlAlgorithm):

    def __init__(self, env, cfg):
        set_composite_lp_aggregate(False).set()
        super().__init__(env, cfg)

    def build(self):
        policy_net = MultiAgentMLP(
            n_agent_inputs=self.env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=self.env.full_action_spec[self.env.action_key].shape[-1],
            n_agents=self.n_agents,
            centralized=False,  # independent actors
            share_params=False,
            device=self.cfg.device,
            depth=self.cfg.arch.depth,
            num_cells=self.cfg.arch.num_cells,
            activation_class=torch.nn.Tanh,
        )
        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )
        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec_unbatched,
            in_keys=[("agents", "logits")],
            out_keys=[("agents", "action")],
            distribution_class=torch.distributions.OneHotCategorical,
            return_log_prob=True,
        )

        # q value net (critric)
        qvalue_net = MultiAgentMLP(
            n_agent_inputs=self.env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2,  # one for each choice
            n_agents=self.n_agents,
            centralized=self.cfg.sac.centralized_qvalue,
            share_params=False,
            device=self.cfg.device,
            depth=self.cfg.arch.depth,
            activation_class=torch.nn.Tanh,
        )
        self.qvalue = TensorDictModule(
            module=qvalue_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action_value")],
        )

    def train(self, logger = None, checkpoint_iter = None):
        self.build_collector(self.policy)

        ## loss module
        self.loss_module = DiscreteSACLoss(
            actor_network=self.policy,
            qvalue_network=self.qvalue,
            action_space=self.env.action_spec_unbatched,
            num_actions=2,
            num_qvalue_nets=2,
        )
        target_updater = SoftUpdate(self.loss_module, eps=self.cfg.optim.loss_eps)

        # TODO: debug
        print("loss target q", self.loss_module.target_qvalue_network_params)
        print("loss q", self.loss_module.qvalue_network_params)


        self.loss_module.set_keys(
            reward=self.env.reward_key,
            action=self.env.action_key,
            action_value=("agents", "action_value"),
            # value=("agents", "action_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=self.cfg.optim.lr)

        pbar = tqdm(total=self.cfg.train.n_iters)

        iter_counter = 0
        for td in self.collector:  # equivalent to `for each batch`
            iter_counter += 1
            td.set(
                ("next", "agents", "done"),
                td.get(("next", "done"))
                .unsqueeze(-1)
                .expand(td.get_item_shape(("next", self.env.reward_key))),
            )
            td.set(
                ("next", "agents", "terminated"),
                td.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(td.get_item_shape(("next", self.env.reward_key))),
            )

            data_view = td.reshape(-1)
            self.replay_buffer.extend(data_view)

            for _ in range(self.cfg.train.n_epochs):
                for _ in range(self.cfg.train.frames_per_batch // self.cfg.train.minibatch_size):
                    subdata = self.replay_buffer.sample()                    
                    loss_vals = self.loss_module(subdata)

                    loss_value = (
                        loss_vals["loss_actor"]
                        + loss_vals["loss_qvalue"]
                        + loss_vals["loss_alpha"]
                    )

                    if logger:
                        prefix = "step"
                        logger.log_scalar(
                            f"{prefix}/loss_actor",
                            loss_vals["loss_actor"].item(),
                        )
                        logger.log_scalar(
                            f"{prefix}/loss_qvalue",
                            loss_vals["loss_qvalue"].item(),
                        )
                        logger.log_scalar(
                            f"{prefix}/loss_alpha",
                            loss_vals["loss_alpha"].item(),
                        )
                        logger.log_scalar(
                            f"{prefix}/total_loss",
                            loss_value.item(),
                        )

                    loss_value.backward()  # compute gradient of current tensor
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.cfg.optim.max_grad_norm
                    )  # Optional
    
                    self.optimizer.step()  # single opimization step
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
            qvalue_path = os.path.join(log_dir, f"checkpoint_qvalue_{suffix}.pt2")
            torch.save(self.policy.state_dict(), policy_path)
            torch.save(self.qvalue.state_dict(), qvalue_path)

    def load(self, log_dir, suffix = None):
            policy_path = os.path.join(log_dir, f"checkpoint_policy_{suffix}.pt2")
            qvalue_path = os.path.join(log_dir, f"checkpoint_qvalue_{suffix}.pt2")
            self.policy.load_state_dict(torch.load(policy_path))
            self.policy.eval()
            self.qvalue.load_state_dict(torch.load(qvalue_path))
            self.qvalue.eval()
    

if __name__ == "__main__":
    pass
