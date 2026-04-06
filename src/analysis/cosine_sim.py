"""
File created at: 2026-04-06 11:57:05
Author: Sam Ghalayini
backdoor-safety-layers/src/analysis/cosine_sim.py
compute + plot cosine similarity 
"""
#%% imports 
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# conduct three different analysis of pairs to look at
# normal to normal pairs, randomly select 
# malicious to malicious
# normal to malicious 
def cosine_sim_list(acts1: np.ndarray, acts2: np.ndarray) -> np.ndarray:
    """returns (num_layers,) average cosine sim per layer for two activations"""
    t1 = t.tensor(acts1).unsqueeze(0)
    t2 = t.tensor(acts2).unsqueeze(0)
    n_layers = t1.shape[1]
    sims = t.zeros(n_layers)
    for i in range(n_layers):
        sims[i] = t.nn.functional.cosine_similarity(t1[:, i, :],t2[:, i, :], dim=1)
    return sims.numpy()

def cosine_pairs(normal, malicious, r:int=500) -> dict[str, np.ndarray]:
    """ for r sets of pairs between N-N, M-M, and N-M"""
    # for r times, take two random activations, compute the sim, then add it to a list of (num_layers,) then divide by r for mean
    # return dict of NN, MM, NM : (num_layers,) which holds the average cosine sim per layer to be plotted
    cosine_dict = {
        "NN": np.zeros((normal.shape[1])),
        "MM": np.zeros((normal.shape[1])),
        "NM": np.zeros((normal.shape[1])),
    }
    for i in range(r):
        n_idx = np.random.choice(normal.shape[0], 3, replace=False)
        m_idx = np.random.choice(malicious.shape[0], 3, replace=False)
        n_samples = normal[n_idx]
        m_samples = malicious[m_idx]
        cosine_dict["NN"] += cosine_sim_list(n_samples[0], n_samples[1])
        cosine_dict["MM"] += cosine_sim_list(m_samples[0], m_samples[1])
        cosine_dict["NM"] += cosine_sim_list(n_samples[2], m_samples[2])

    cosine_dict["NN"] = cosine_dict["NN"] / r
    cosine_dict["MM"] = cosine_dict["MM"] / r
    cosine_dict["NM"] = cosine_dict["NM"] / r
    return cosine_dict

def plot_cosine_sims(model_dir:Path, cosine_dict: dict[str, np.ndarray]):
    # 1x3 subplot per dict with std and nice labels
    fig, axes = plt.subplots(1,3, figsize=(15,4), sharey=True)
    for ax, (label, sims) in zip(axes, cosine_dict.items()):
        ax.plot(sims)
        ax.set_title(label)
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Cosine Similarity")
    plt.suptitle(str(model_dir).split("/")[-1])
    plt.tight_layout(rect=[0,0,1,0.99]) # type: ignore
    # save fig
    project_root = Path(__file__).parent.parent.parent
    fig_dir = project_root / "figures" / "cosine_gap"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"cosine_gap_{str(model_dir).split('/')[-1]}.png", dpi=150)
    plt.show()

# %% run main
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    for model_dir in sorted((project_root / "data" / "activations").iterdir()):
        normal = np.load(model_dir / "normal_activations.npy")
        malicious = np.load(model_dir / "malicious_activations.npy")
        cosine_dict = cosine_pairs(normal, malicious)
        plot_cosine_sims(model_dir, cosine_dict)

# %%
