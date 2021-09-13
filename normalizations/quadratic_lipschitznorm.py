import torch, torch.nn as nn
from torch_scatter import scatter


class Quadratic_LipschitzNorm(nn.Module):
    def __init__(self, eps = 1e-12):
        super(Quadratic_LipschitzNorm, self).__init__()
        self.eps = eps

    def forward(self, Q, K, V, alpha, index):
        
        Q_F = torch.norm(Q) # frobenious norm of Q
        K_2 = torch.norm(K,dim = -1)
        V_2 = torch.norm(V,dim = -1)
        K_inf_2 = scatter(src= K_2, index = index, dim=0, reduce = 'max')
        V_inf_2 = scatter(src= V_2, index = index, dim=0, reduce = 'max')
        
        uv = Q_F * K_inf_2 
        uw = Q_F * V_inf_2
        vw = K_inf_2 * V_inf_2

        max_over_norms = torch.max(torch.stack([uv,uw,vw]),dim=0).values 
        alpha = alpha / (max_over_norms[index] + self.eps)
        # dsdas
        
        return alpha