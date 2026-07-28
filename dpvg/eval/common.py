# a class that analyze from rollout


class Simulator():
    def __init__(self):
        pass

    def rollout_and_save(self, n_episodes: int, path: str):
        pass

    def load(self):
        pass


if __name__ == "__main__":
    from pprint import pprint
    from tqdm import tqdm
    import pickle

    from dpvg.envs.voting_game import *
    from dpvg.algorithms.common import episode_reward_ext
    from dpvg.envs.static_policies import batch_greedy_policy, batch_gini_policy
    

    # env config and spec checking
    n_envs = 10
    n_episodes = 10_000
    ep_max_steps = 100

    env_config = SimpleDpvgConfig()
    env = get_vectorized_sdpvg(
        env_config=env_config,
        n_workers=n_envs,
        mode="p",
        flatten_obs=True,
    )
    env = episode_reward_ext(env)
    env.n_agents = env_config.n_agents
    env.check_env_specs()
    print(env)

    collection = []

    for _ in tqdm(range(n_episodes // n_envs), position=0):

        # gini
        env.reset()
        episodes = env.rollout(
            max_steps=ep_max_steps,
            policy=batch_greedy_policy,
            break_when_any_done=False,  # prevent early termination
        )
        # print(episodes.shape)

        for i in tqdm(range(episodes.shape[0]), position=1, leave=False):
            ep = episodes[i]
            # print(ep.shape)

            ep_done: Tensor = ep.get(("next", "done")).squeeze(-1)
            id_done = ep_done.nonzero()

            if id_done.numel() == 0:
                id_done = ep_max_steps - 1
            else:
                id_done = id_done[0]

            # print(ep)

            ep_collection = []

            for j in range(id_done+1):
                ep_collection.append(get_step_detail(ep, j))

            collection.append(ep_collection)

    print(len(collection))

    with open("outs/greedy_10000max100_default.pkl", "wb") as f:
        pickle.dump(collection, f)

    # final call
    env.close()
