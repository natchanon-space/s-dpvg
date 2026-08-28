import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import pandas as pd


def read_collection(path: str, n_episodes: int = 5000):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return raw[:n_episodes]

def ep_len(collection: list):
    ep = []
    for col in collection:
        last_step_info = col[-1]["info"]
        ep.append(last_step_info["episode_length"][0])
    print(np.mean(ep))
    return ep

# models = [
#     # "greedy",
#     # "gini",
#     "mappo",
#     "ippo",
#     "qmix",
# ]
# for m in tqdm(models):
#     mean = []
#     for i in range(100, 1001, 100):
#         mean.append(np.mean(ep_len(read_collection(f"outs/default_{m}_{i}.pkl"))))
#     plt.plot(mean, label=m)


def overlapping_ep_len(model: str):
    i = [j for j in range(100, 1001, 100)]
    mean = []
    for j in range(100, 1001, 100):
        mean.append(ep_len(read_collection(f"outs/default_{model}_{j}.pkl")))

    df = pd.DataFrame(dict(iteration=i, length=mean))
    # offset = df["iteration"].map(ord)
    df["length"] += df["iteration"]

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="iteration", hue="iteration", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    # g.map(sns.kdeplot, "length",
    #     bw_adjust=.5, clip_on=False,
    #     fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "length", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "length")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

# greedy = read_collection("outs/default_greedy.pkl")
# gini = read_collection("outs/default_gini.pkl")
# mappo = read_collection("outs/default_mappo_1000.pkl")
# ippo = read_collection("outs/default_ippo_1000.pkl")
# qmix = read_collection("outs/default_qmix_1000.pkl")
# masac = read_collection("outs/default_masac_1000.pkl")

# plt.hist(ep_len(greedy), bins=64, range=(0, 256), label="greedy", a="")
# plt.hist(ep_len(gini), bins=64, range=(0, 256), label="gini")
# plt.hist(ep_len(mappo), bins=64, range=(0, 256), label="mappo", alpha=0.5)
# plt.hist(ep_len(ippo), bins=64, range=(0, 256), label="ippo", alpha=0.5)
# plt.hist(ep_len(qmix), bins=64, range=(0, 256), label="qmix", alpha=0.5)
# plt.hist(ep_len(masac), bins=64, range=(0, 256), label="masac", alpha=0.5)

# print(stats.ks_2samp(ep_len(mappo), ep_len(ippo)))

overlapping_ep_len("mappo")

plt.legend()
plt.show()