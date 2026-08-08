import matplotlib.pyplot as plt
import pickle
import numpy as np


def read_collection(path: str, n_episodes: int = 5000):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return raw[:n_episodes]

def ep_len(collection: list):
    ep = []
    for col in collection:
        last_step_info = col[-1]["info"]
        ep.append(last_step_info["episode_length"][0])
    return ep

greedy = read_collection("outs/default_greedy.pkl")
gini = read_collection("outs/default_gini.pkl")
mappo = read_collection("outs/default_mappo_1000.pkl")


plt.hist(ep_len(greedy), bins=64, range=(0, 256), label="greedy", a="")
plt.hist(ep_len(gini), bins=64, range=(0, 256), label="gini")
plt.hist(ep_len(mappo), bins=64, range=(0, 256), label="mappo")
plt.legend()
plt.show()