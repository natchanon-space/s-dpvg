from torchrl.record import CSVLogger
from torchrl.envs import check_env_specs
import yaml
from tensordict import TensorDict

from dpvg.envs.voting_game import SimpleDpvgConfig, get_vectorized_sdpvg
from dpvg.algorithms.common import episode_reward_ext
from dpvg.algorithms.mappo import Mappo
from dpvg.algorithms.sac import SAC
from dpvg.algorithms.qmix import Qmix
from configs.common import dict_to_namespace


MODEL_NAME = "qmix"
MODEL_PATH = f"logs/default_{MODEL_NAME}"
MODEL_CONFIG = f"configs/default_{MODEL_NAME}.yaml"


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

    # define batched env
    env = get_vectorized_sdpvg(
        env_config=env_config,
        n_workers=n_envs,
        mode="p",
        flatten_obs=True,
    )
    env = episode_reward_ext(env)
    env.n_agents = env_config.n_agents

    # read model config
    with open(MODEL_CONFIG, "r") as f:
        cfg = dict_to_namespace(yaml.safe_load(f))

    model = Qmix(env, cfg)
    model.load(MODEL_PATH, 1000)

    td: TensorDict = model.policy(env.reset())

    # print(ippo.policy(env.reset()))
    print(td.get(('agents', "action")))

    env.close()
