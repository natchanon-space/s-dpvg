from abc import ABC, abstractmethod
from torchrl.envs import EnvBase, RewardSum, TransformedEnv
from torchrl.record import CSVLogger
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from tensordict.nn import TensorDictModule


## environment extensions section
def episode_reward_ext(env: EnvBase, out_keys=[("agents", "episode_reward")]) -> EnvBase:
    return TransformedEnv(
        env,
        RewardSum(
            in_keys=[env.reward_key],
            out_keys=out_keys
        )
    )


class MarlAlgorithm(ABC):
    def __init__(self, env: EnvBase, cfg):
        self.env = env
        self.n_agents = self.env.n_agents
        self.env.reset()
        self.cfg = cfg
        self.build()

    def build(self):
        pass

    def build_collector(self, policy: TensorDictModule):
        self.collector = Collector(
            self.env,
            policy,
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
        

    def train(self, logger: CSVLogger = None, checkpoint_iter: int = None):
        pass

    def save(self, log_dir: str, suffix: str = None):
        pass

    def load(self, log_dir: str, suffix: str = None):
        pass

    def close(self):
        pass
