from dpvg.envs.static_policies import batch_greedy_policy, batch_gini_policy
from dpvg.envs.voting_game import *
from dpvg.algorithms.common import episode_reward_ext
from dpvg.eval.common import Simulator


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

    # collections
    simulator = Simulator(env, n_envs, max_steps)

    # collect gini
    simulator.rollout_and_save(
        policy=batch_gini_policy,
        n_episodes=n_episodes,
        path="outs/default_gini.pkl"
    )

    # collect greedy
    simulator.rollout_and_save(
        policy=batch_greedy_policy,
        n_episodes=n_episodes,
        path="outs/default_greedy.pkl"
    )

    env.close()
