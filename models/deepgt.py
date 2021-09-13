########################################################
# DeepGraph Transformer for Node Classification
########################################################


# import libraries 
from normalizations.quadratic_lipschitznorm import Quadratic_LipschitzNorm
from typing import Union, Tuple, Optional
# from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from torch_geometric.typing import PairTensor, Adj, OptTensor
import math
from torch import Tensor
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.norm import PairNorm, LayerNorm
from torch_scatter import scatter


class DeepTransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:
    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via
            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}
            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{x}_i, \mathbf{m}_i, \mathbf{x}_i - \mathbf{m}_i ])`
            (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:
            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),
            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:
            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)
            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, norm = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(DeepTransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.norm = norm

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):
        """"""

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)

        # LipschitzNorm
        if self.norm is not None:
            alpha = self.norm(query, key, value, alpha, index)
        

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_attr is not None:
            out = value + edge_attr
        else:
            out = value + 0

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)





class DeepGT(nn.Module):
    def __init__(self, idim, hdim, odim, num_layers, heads=1, norm = None, beta =False, ogb = False):

        super(DeepGT, self).__init__()
        self.num_layers = num_layers
        self.norm_name = norm
        self.layernorm = None
        self.pairnorm = None
        self.norm = None
        self.ogb = ogb

        if self.norm_name == "pairnorm":
            self.pairnorm = PairNorm() 
        elif self.norm_name == "pairnorm-si":
            self.pairnorm = PairNorm(scale_individually=True)
        elif self.norm_name == "lipschitznorm":
            self.norm = Quadratic_LipschitzNorm()
        elif self.norm_name == "lipschitznorm+layernorm":
            self.norm = Quadratic_LipschitzNorm()
            self.layernorm = LayerNorm(hdim)
        elif self.norm_name == "layernorm":
            self.layernorm = LayerNorm(hdim)


        self.layers = nn.ModuleList([DeepTransformerConv(hdim, hdim, heads=heads, dropout=0.3, norm=self.norm, beta = beta)])
        for _ in range(1,num_layers):
            self.layers.append(DeepTransformerConv(heads * hdim, hdim, heads=heads, dropout=0.3, norm=self.norm, beta = beta))

        self.lin = nn.Linear(idim, hdim)
        self.fc1 = nn.Linear(heads * hdim, odim)


        
    def forward(self, batched_data):

        if self.ogb:
            x, edge_index = batched_data.x, batched_data.adj_t
        else:
            x, edge_index = batched_data.x, batched_data.edge_index

        # x, edge_index = batched_data.x, batched_data.edge_index
        h = self.lin(x)
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, p=0.5, training = self.training)
            h = layer(h, edge_index)
            if self.layernorm is not None:
                h = self.layernorm(h)
            if i < self.num_layers - 1:
                h = F.elu(h)
            if self.pairnorm is not None:
                h = self.pairnorm(h)

            
            
        return F.log_softmax(self.fc1(h), dim=1)
