from typing import Dict, Any
from .network.nn import DenoisingGraphTransformer
from .diffusion import PermutableGaussianDiffusion
from ..data import Kinematic


def num_graph_nodes(skeleton) -> int:
    """Number of graph nodes of a skeleton: bones for the bone direction representation, joints otherwise."""
    return (skeleton.bone_adj_matrix() if skeleton.node_type == "bone" else skeleton.adj_matrix).shape[-1]


class DiffusionManager():
    """Builds the denoiser and the diffusion process for the skeletons (datasets) seen during training."""

    def __init__(self, skeletons: Dict[str, Kinematic], num_nodes: int,
                 diffusion_type: str = 'PermutableGaussianDiffusion', **kwargs):
        assert diffusion_type == 'PermutableGaussianDiffusion', diffusion_type
        model = self.get_network(num_nodes=num_nodes, **kwargs)
        self.diffusion = PermutableGaussianDiffusion(
            num_nodes_per_dataset={name: num_graph_nodes(skel) for name, skel in skeletons.items()},
            num_nodes=num_nodes, model=model, **kwargs)

    def get_diffusion(self):
        return self.diffusion

    def get_network(self, latent_size=96, diffusion_arch: Dict[str, Any] = {}, diffusion_conditioning=True, **kwargs):
        assert diffusion_conditioning, "the denoiser is always conditioned on the observed motion"
        return DenoisingGraphTransformer(dim=latent_size, cond_dim=latent_size, out_dim=latent_size,
                                         **diffusion_arch, **kwargs)
