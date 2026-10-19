from typing import Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedLinear2(nn.Module):
    def __init__(self, in_features, out_features, activation=None, bias=False):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.ones(out_features, in_features))
        self.bias = bias

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            self.bias_w = torch.nn.Parameter(torch.zeros(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_w, -bound, bound)

    def forward(self, input, adj: torch.Tensor = None, mask: torch.Tensor = None):
        # `input` is of shape (batch_size, n, n)
        # `mask` is of shape (batch_size, n,n) as the weigths


        w = self.weight
        input = input@w.mT
        if self.bias:
            if mask is None:
                mask = (adj.sum(-1) != 0).unsqueeze(-1)
            input = (input + self.bias_w)* mask
        return input if self.activation is None else self.activation(input)
