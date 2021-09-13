import torch, torch.nn as nn
from torch_scatter import scatter

class LipschitzNorm(nn.Module):
    def __init__(self, att_norm = 4, recenter = False, scale_individually = True, eps = 1e-12):
        super(LipschitzNorm, self).__init__()
        self.att_norm = att_norm
        self.eps = eps
        self.recenter = recenter
        self.scale_individually = scale_individually

    def forward(self, x, att, alpha, index):
        att_l, att_r = att
        
        if self.recenter:
            mean = scatter(src = x, index = index, dim=0, reduce='mean')
            x = x - mean


        norm_x = torch.norm(x, dim=-1) ** 2
        max_norm = scatter(src = norm_x, index = index, dim=0, reduce = 'max').view(-1,1)
        max_norm = torch.sqrt(max_norm[index] + norm_x)  # simulation of max_j ||x_j||^2 + ||x_i||^2

        
        # scaling_factor =  4 * norm_att , where att = [ att_l | att_r ]         
        if self.scale_individually == False:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim = -1))
        else:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim=-1), dim = -1)

        alpha = alpha / ( norm_att * max_norm + self.eps )
        return alpha