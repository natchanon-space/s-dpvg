from abc import ABC, abstractmethod
from torchrl.envs import EnvBase, RewardSum, TransformedEnv
from torchrl.record import CSVLogger


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

    def build():
        pass

    def train(self, logger: CSVLogger = None, checkpoint_iter: int = None):
        pass

    def save(self, log_dir: str, suffix: str = None):
        pass

    def load(self, log_dir: str, suffix: str = None):
        pass

    def close(self):
        pass
