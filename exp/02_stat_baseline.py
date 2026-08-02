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

    plt.hist(ep_length, bins=20)
    # plt.show()


n_episodes = 5000

# with open("outs/default_gini.pkl", "rb") as f:
#     collections = pickle.load(f)

# sum_len = 0

# for i in tqdm(range(n_episodes)):
#     ep = collections[i]
#     sum_len += ep[-1]["info"]["episode_length"][0]

# print(sum_len / n_episodes)

read_collection("outs/default_greedy.pkl")
read_collection("outs/default_gini.pkl")

plt.show()
