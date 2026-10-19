from typing import List, Tuple, Optional
import torch

def get_adj_matrix(limbseq:List[Tuple[int]], num_nodes: Optional[int]=None):
    if num_nodes is None:
        assert 0, "Not tested, it is probably bugged"
        num_nodes = max([max(l) for l in limbseq])
    
    adj = torch.zeros((num_nodes, num_nodes))
    for i, j in limbseq:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def plot_matrix(matrix, node_names):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    Sigma_N = matrix.cpu().clone().numpy()
    color = 'Purples'
    cmap = matplotlib.colormaps[color].set_bad("white")
    # cmap[0] = 

    fig, ax = plt.subplots(1,1, figsize=(6, 6),sharex=True,  subplot_kw=dict(box_aspect=1),)
    vmax = Sigma_N.max()
    Sigma_N[Sigma_N <=0.0000] = np.nan
    im = ax.imshow(Sigma_N, cmap=color, vmin=0., vmax=vmax)
    ax.set_xticks(np.arange(len(Sigma_N)))
    ax.set_xticklabels(labels=node_names, rotation=45, ha="right",
            rotation_mode="anchor")
    ax.set_yticks(np.arange(len(Sigma_N)))
    ax.set_yticklabels(labels=node_names, rotation=45, ha="right",
            rotation_mode="anchor")
    fig.colorbar(im, cmap=cmap)
    plt.show()
    fig.savefig("./today.pdf", format="pdf", bbox_inches="tight")