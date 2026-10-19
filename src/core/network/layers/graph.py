from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import get_norm_layer, SequentialWithAdj


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, *args, bias: bool = True, num_nodes: int = None, graph_influence: Union[torch.Tensor, nn.Parameter] = None,
                 learn_influence: bool = False, learn_influence_via_hypernetwork=False, hypernet_layer_sizes: List[int] = [0], **kwargs):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param num_nodes: Number of nodes.
        :param graph_influence: Graph Influence Matrix
        :param learn_influence: If set to ``False``, the layer will not learn an the Graph Influence Matrix.
        :param bias: If set to ``False``, the layer will not learn an additive bias.

        with learn_influence=True, 
        the layer performs GCN for k=1 (first order neighbours) without final activation function.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Feature aggregation
        self.G = nn.Parameter(torch.Tensor(
            self.in_features, self.out_features))
        nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))

        self.learn_influence = learn_influence
        self.learn_influence_via_hypernetwork = learn_influence_via_hypernetwork

        if self.learn_influence_via_hypernetwork:
            assert self.learn_influence, "If learning influence via hypernetwork, influence matrix must be learnable."

        if graph_influence is not None:
            assert num_nodes == graph_influence.shape[
                0] or num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
            num_nodes = graph_influence.shape[0]
            if type(graph_influence) is nn.Parameter:
                assert learn_influence, "Graph Influence Matrix is a nn.Parameter, therefore it must be learnable."
                self.aggr = graph_influence
            elif learn_influence:
                self.aggr = nn.Parameter(graph_influence)
            else:
                self.register_buffer('G', graph_influence)

            Gmask = torch.zeros_like(self.G)
            Gmask[self.aggr != 0] = 1
            for i in range(num_nodes):
                Gmask[i, i] = 1  # Set diagonal to 1
            self.register_buffer('Gmask', Gmask)

        # Feature extraction
        self.weight = nn.Parameter(torch.Tensor(
            self.out_features, self.in_features))
        self.mm = torch.matmul

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def get_influence(self, adj: Optional[torch.Tensor] = None):
        if self.learn_influence:
            if not self.learn_influence_via_hypernetwork:
                ...


                # Lambda, U = torch.linalg.eigh(l_norm, UPLO='L')
                # Lambda[~mask_eig] = 0
                # eig_pow_n[~mask_eig] = 0
                # # How to treat weights of zeros eigenvalues? actually they should all be reordered, right?
            assert 0, "Not implemented yet"
            assert adj is not None, "Hypernetwork needs input."
            aggr = self.hypernetwork(adj)
            aggr = torch.nn.functional.normalize(aggr, p=1., dim=-1)
        else:
            aggr = torch.nn.functional.normalize(adj, p=1., dim=-1)
            return aggr

    def get_G(self, adj: Optional[torch.Tensor] = None):
        g = self.G
        return g

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, adj: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        w = self.weight

        # feature extraction
        output = self.mm(input, w.transpose(-2, -1))
        # the masking is obtained by the adjacency matrix
        if self.bias is not None:
            if mask is None:
                mask = (adj.sum(-1) != 0).unsqueeze(-1)
            output = (output + self.bias) * mask

        # aggregation
        g = self.get_G(adj)
        output = (self.get_influence(adj)@input)@g + output

        return output


class Block(nn.Module):
    def __init__(self, dim, dim_out, norm_type='none', act_type='tanh', layer_type: nn.Module = GraphConv, *args, **kwargs):
        super().__init__()
        self.proj = SequentialWithAdj(layer_type(dim, dim_out, *args, **kwargs))

        self.norm = get_norm_layer(norm_type, dim_out)
        if act_type == 'tanh':
            self.act = nn.Tanh()
        else:
            assert 0, f"Activation type {act_type} not implemented!"

    def forward(self, x, scale_shift=None, adj: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, attn_bias: Optional[torch.Tensor] = None):
        x = self.proj(x, adj=adj, mask=mask, attn_bias=attn_bias)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
            if mask is None:
                mask = (adj.sum(-1) != 0).unsqueeze(-1)
            x = x * mask

        x = self.act(x)
        return x
