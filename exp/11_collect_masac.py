import yaml

from dpvg.envs.voting_game import *
from dpvg.algorithms.common import episode_reward_ext
from dpvg.algorithms.sac import SAC
from dpvg.eval.common import Simulator
from configs.common import dict_to_namespace


# config
n_episodes = 5000

frames_per_batch = 4096
max_steps = 256
n_envs = frames_per_batch // max_steps

if __name__ == "__main__":
    # env settings
    env_config = SimpleDpvgConfig(
        n_agents=5,
        k=3,
        l=6,
        passive=0,
        max_steps=max_steps
    )
    env = get_vectorized_sdpvg(
        env_config=env_config,
        n_workers=n_envs,
        mode="p",
        flatten_obs=True
    )
    env = episode_reward_ext(env)
    env.n_agents = env_config.n_agents
    env.check_env_specs()

    # mappo setup
    with open("configs/default_masac.yaml", "r") as f:
            cfg = dict_to_namespace(yaml.safe_load(f))
    masac = SAC(env, cfg)

    for i in range(100, 1001, 100):
        masac.load("logs/default_masac", f"{i}")

        # collections
        simulator = Simulator(env, n_envs, max_steps)

        # collect gini
        simulator.rollout_and_save(
            policy=masac.policy,
            n_episodes=n_episodes,
            path=f"outs/default_masac_{i}.pkl"
        )

    env.close()
