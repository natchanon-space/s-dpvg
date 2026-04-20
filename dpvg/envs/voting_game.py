from dataclasses import dataclass
import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Binary, Composite, Bounded, OneHot, Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.batched_envs import ParallelEnv, SerialEnv
from torchrl.envs.transforms import TransformedEnv, Compose, FlattenObservation, CatTensors, RenameTransform


def repeat_batch_dim(t: torch.Tensor, times: int):
    return t.unsqueeze(0).repeat([times] + [1]*len(t.shape))


@dataclass
class SimpleDpvgConfig():
    n_agents: int = 5 # number of agents, must be odd positive
    k: int = 3          # choice sampled from U[0, k]
    l: int = 10         # terminal condition is (max - min > l)
    passive: int = 0    # reward shaping if applicable!
    dtype = torch.float32


class SimpleDpvgEnv(EnvBase):

    def __init__(self, *, env_config: SimpleDpvgConfig = SimpleDpvgConfig(), device = None, batch_size = None, run_type_checks = False, allow_done_after_reset = False, spec_locked = True, auto_reset = False):
        super().__init__(device=device, batch_size=[], run_type_checks=run_type_checks, allow_done_after_reset=allow_done_after_reset, spec_locked=spec_locked, auto_reset=auto_reset)
        # self.batch_size = [env_config.n_agents]
        # read config
        self.n_agents = env_config.n_agents
        self.k = env_config.k
        self.l = env_config.l
        self.passive = env_config.passive
        self.dtype = env_config.dtype
        # initiate spec and initial states
        self.agent_ids = torch.eye(self.n_agents, dtype=self.dtype)
        self.current_choices = None
        self.raw_power = None
        self._make_spec()
        self._reset(None)

    def _reset(self, tensordict, **kwargs):
        """return the first tensordict of a rollout."""
        # reset stages
        self.raw_power = torch.zeros(self.n_agents, dtype=self.dtype)
        self.current_choices = self._rand_choices()
        # create initial rollout return
        next_td = TensorDict(
            {
                "agents": {
                    "observation": {
                        "id": self.agent_ids,
                        "state": repeat_batch_dim(self.raw_power, self.n_agents),
                        "choices": repeat_batch_dim(self.current_choices, self.n_agents),
                    }
                }
            },
        )
        return next_td

    def _step(self, tensordict):
        # counting vote
        votes = tensordict["agents"]["action"]
        count_1 = torch.sum(votes)
        count_0 = self.n_agents - count_1
        win_choice_idx = int(count_1 > count_0) 
        # save power gain as reward
        power_gain = self.current_choices[:, win_choice_idx]
        # transition to next state
        self.raw_power = self.raw_power + power_gain
        next_state = self.raw_power - torch.min(self.raw_power)
        # generate new choices
        self.current_choices = self._rand_choices()
        # combine output
        next_tensordict = TensorDict(
            {
                "agents": {
                    "observation": {
                        "id": self.agent_ids,
                        "state": repeat_batch_dim(next_state, self.n_agents),
                        "choices": repeat_batch_dim(self.current_choices, self.n_agents),
                    },
                    "reward": power_gain.unsqueeze(-1),
                },
                "done": (torch.max(self.raw_power) - torch.min(self.raw_power) > self.l).unsqueeze(0)  # shape [1]
            }
        )
        return next_tensordict

    def _set_seed(self, seed):
        return super()._set_seed(seed)

    def _rand_choices(self):
        choice_pair_spec = Bounded(
            low=0,
            high=self.k+1,
            shape=(self.n_agents, 2),
            dtype=torch.int32
        )
        choices = choice_pair_spec.rand()
        if not (choices[:, 0] != choices[:, 1]).all():
            return self._rand_choices()
        return choices.type(self.dtype)

    def _make_spec(self):
        self.observation_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Composite(
                            {
                                "id": OneHot(
                                    n=self.n_agents,
                                    shape=(self.n_agents, self.n_agents),
                                    dtype=self.dtype,
                                ),
                                "state": Bounded(
                                    low=0,
                                    high=self.l + self.k + 1,
                                    shape=(self.n_agents, self.n_agents),
                                    dtype=self.dtype,
                                ),
                                "choices": Bounded(
                                    low=0,
                                    high=self.k+1,
                                    shape=(self.n_agents, self.n_agents, 2),
                                    dtype=self.dtype,
                                )
                            }
                        )
                    },
                    shape=(self.n_agents,)
                ),
            }
        )
        self.action_spec = Composite(
            {
                "agents": Composite(
                    {
                        "action": Binary(
                            n=1,
                            shape=(self.n_agents, 1)
                        )
                    },
                    shape=(self.n_agents,)
                ),
            },
        )
        # reward for each agents is in range of [0, k+passive+1)
        self.reward_spec = Composite(
            {
                "agents": Composite(
                    {
                        "reward": Unbounded(
                            dtype=torch.float32,
                            shape=(self.n_agents, 1)
                        )
                    },
                    shape=(self.n_agents,),
                )
            }
        )
        self.done_spec = Binary(
            n=1,
            shape=(1,),
            dtype=torch.bool,
        )
        # do not have to define done_spec, we will use default one


def get_vectorized_sdpvg(env_config: SimpleDpvgConfig, n_workers: int, mode="p", flatten_obs=True, device=None):

    def _make_env():
        return SimpleDpvgEnv(env_config=env_config, device=device)
    
    match mode:
        case "p":
            env = ParallelEnv(
                num_workers=n_workers,
                create_env_fn=_make_env
            )
        case "s":
            env = SerialEnv(
                num_workers=n_workers,
                create_env_fn=_make_env
            )
        case _:
            assert ValueError("Invalide environment mode: either (p)arallel or (s)erial")

    if flatten_obs:
        env = TransformedEnv(
            env,
            Compose(
                FlattenObservation(
                    first_dim=-2,
                    last_dim=-1,
                    in_keys=[("agents", "observation", "choices")]
                ),
                CatTensors(
                    in_keys=[
                        ("agents", "observation", "choices"),
                        ("agents", "observation", "id"),
                        ("agents", "observation", "state"),
                    ],
                    # out_key=("agents", "observation"),
                    del_keys=True
                ),
                RenameTransform(
                    in_keys=("observation_vector"),
                    out_keys=("agents", "observation")
                )
            )
        )

    return env
