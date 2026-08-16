from torchrl.record import CSVLogger
from torchrl.envs import check_env_specs
import yaml

from dpvg.envs.voting_game import SimpleDpvgConfig, get_vectorized_sdpvg
from dpvg.algorithms.common import episode_reward_ext
from dpvg.algorithms.qmix import Qmix
from configs.common import dict_to_namespace


if __name__ == "__main__":
    # env config
    env_config = SimpleDpvgConfig(
        n_agents=5,
        k=3,
        l=6,
        max_steps=256
    )

    # calculate appropriate number of envs
    frame_per_batch = 4096
    n_envs = frame_per_batch // env_config.max_steps

    # logger
    logger = CSVLogger(
        exp_name="default_qmix",
        log_dir="logs",
    )

    # define batched env
    env = get_vectorized_sdpvg(
        env_config=env_config,
        n_workers=n_envs,
        mode="p",
        flatten_obs=True,
    )
    env = episode_reward_ext(env)
    env.n_agents = env_config.n_agents
    check_env_specs(env)
    env.reset()

    # read model config
    with open("configs/default_qmix.yaml", "r") as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    qmix = Qmix(env, cfg)
    qmix.train(logger, checkpoint_iter=100)

    qmix.close()
