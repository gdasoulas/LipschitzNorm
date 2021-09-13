from torch_scatter import scatter_add, scatter_mean
import numpy as np, torch.nn as nn, torch


class PairNorm(nn.Module):
    r"""Applies PairNorm normalization layer over aa batch of nodes
    """
    def __init__(self, s = 1):
        super(PairNorm, self).__init__()
        self.s = s

    def forward(self, x, batch=None):

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_c = x - scatter_mean(x, batch, dim=0)[batch]
        out = self.s * x_c / scatter_mean((x_c * x_c).sum(dim=-1, keepdim=True),
                             batch, dim=0).sqrt()[batch]

        return out
