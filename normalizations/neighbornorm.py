import torch, torch.nn as nn
from torch_scatter import scatter

class NeighborNorm(nn.Module):
    def __init__(self, scale = 1, eps = 1e-12):
        super(NeighborNorm, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x, edge_index):
        index_i, index_j = edge_index[0], edge_index[1]
        mean = scatter(src = x[index_j], index = index_i, dim=0, reduce='mean')
        reduced_x = (x - mean)**2

        # is it 'add' or 'mean'??
        # std = torch.sqrt(scatter(src = reduced_x[edge_index[0]], index = edge_index[1] , dim = dim, reduce = 'add')+ eps) # we add the eps for numerical instabilities (grad of sqrt at 0 gives nan)
        std = torch.sqrt(scatter(src = reduced_x[index_j], index = index_i, dim=0, reduce = 'mean')+ self.eps) # we add the eps for numerical instabilities (grad of sqrt at 0 gives nan)
        out =  self.scale * x / std
        infs = torch.isinf(out)
        out[infs] = x[infs]
        return out 