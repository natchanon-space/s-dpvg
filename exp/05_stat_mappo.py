from pprint import pprint
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import numpy as np


def read_collection(path: str, n_episodes: int = 5000):
    with open(path, "rb") as f:
        collections = pickle.load(f)

    last_step_info = []

    for i in tqdm(range(n_episodes)):
        ep = collections[i]
        last_step_info.append(ep[-1]["info"])

    ep_length = [last["episode_length"][0] for last in last_step_info]

    print(np.mean(ep_length), np.min(ep_length), np.max(ep_length))

    plt.hist(ep_length, bins=64)
    plt.show()


n_episodes = 5000

for i in range(100, 1001, 100):
    read_collection(f"outs/default_mappo_{i}.pkl")
