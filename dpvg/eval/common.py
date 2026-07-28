from tensordict.nn import TensorDictModule


class Simulator():
    def __init__(self, env, n_envs: int, max_steps: int):
        self.env = env
        self.n_envs = n_envs
        self.max_steps = max_steps

    def rollout_and_save(self, policy: TensorDictModule, n_episodes: int, path: str):
        ep_collection = []

        for _ in tqdm(range(n_episodes // self.n_envs), position=0):
            self.env.reset()
            episodes = env.rollout(
                max_steps=self.max_steps,
                policy=policy,
                break_when_any_done=False,  # prevent early termination
            )

            for i in tqdm(range(episodes.shape[0]), position=1, leave=False):
                ep = episodes[i]

                ep_done: Tensor = ep.get(("next", "done")).squeeze(-1)
                id_done = ep_done.nonzero()

                if id_done.numel() == 0:
                    id_done = ep_max_steps - 1
                else:
                    id_done = id_done[0]  # select only first done ep for each batch

                ep_detail = []
                for j in range(id_done+1):
                    ep_detail.append(get_step_detail(ep, j))

                ep_collection.append(ep_collection)

        self.env.close()

        with open(path, "wb") as f:
            pickle.dump(ep_collection, f)

        print(f"{len(ep_collection)} episodes are saved! as {path}")

    def load(self, path: str) -> list[list[dict]]:
        with open(path, "rb") as f:
            ep_collection = pickle.load(f)
        return ep_collection

if __name__ == "__main__":
    from pprint import pprint
    from tqdm import tqdm
    import pickle

    from dpvg.envs.voting_game import *
    from dpvg.algorithms.common import episode_reward_ext
    from dpvg.envs.static_policies import batch_greedy_policy, batch_gini_policy


    # env config and spec checking
    n_envs = 10
    n_episodes = 1000
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

    simulator = Simulator(env, n_envs=n_envs, max_steps=ep_max_steps)

    simulator.rollout_and_save(
        policy=batch_gini_policy,
        n_episodes=n_episodes,
        path="outs/trial_greedy_default.pkl"
    )
