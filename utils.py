import numpy as np, torch.nn as nn, torch
import os.path as osp, os, statistics, json
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from torch_scatter import scatter_mean
from torch_scatter import scatter

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    sort_idx = torch.argsort(labels)
    # print(f'sorted p:{preds[sort_idx]}')
    # print(f'sorted l:{labels[sort_idx]}')
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def safe_mkdir(dir_):
    if not osp.isdir(dir_):
        os.mkdir(dir_)


class Logger(object):
    def __init__(self, log_dir, chkpt_interval=1):
        super(Logger, self).__init__()
        safe_mkdir(os.path.join(log_dir))
        self.log_dir = log_dir
        # self.log_path = os.path.join(log_dir, name, 'logs.json')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
            self.save(model)
        self.logs['epoch'] += 1

    def save(self, model, save_model_params=False):
        with open(os.path.join(self.log_dir, 'logs.json'), 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        if save_model_params:
            torch.save(model.state_dict(), os.path.join(self.log_dir, 'model.pt'))


def safe_double_concat(all_pred_labels, all_target, pred_labels, target):
    if all_pred_labels is None:
        return pred_labels, target
    else:
        return torch.cat((all_pred_labels, pred_labels)), torch.cat((all_target, target))


def get_cv_results(log_dict, k_folds):

    # mean over the folds of the maximum accuracy of all epochs per fold
    # max_acc_train = [max(log_dict['train_acc_' + str(k)]) for k in range(k_folds)]
    # max_acc_test = [max(log_dict['test_acc_' + str(k)]) for k in range(k_folds)]
    # cv_acc_train = (statistics.mean(max_acc_train), statistics.stdev(max_acc_train))
    # cv_acc_test = (statistics.mean(max_acc_test), statistics.stdev(max_acc_test))
    
    metric = "acc"
    max_len = min([len(log_dict['train_'+metric+'_' + str(k)]) for k in range(k_folds)])
    accs_train = [[log_dict['train_'+metric+'_' + str(k)][i] for k in range(k_folds)] for i in range(max_len)]
    mean_acc_train = [statistics.mean(x) for x in accs_train]
    std_acc_train = [statistics.stdev(x) for x in accs_train]
    accs_test = [[log_dict['test_'+metric+'_' + str(k)][i] for k in range(k_folds)] for i in range(max_len)]
    mean_acc_test = [statistics.mean(x) for x in accs_test]
    std_acc_test = [statistics.stdev(x) for x in accs_test]
    
    cv_acc_train = (max(mean_acc_train), std_acc_test[mean_acc_train.index(max(mean_acc_train))])
    cv_acc_test = (max(mean_acc_test), std_acc_test[mean_acc_test.index(max(mean_acc_test))])

    return cv_acc_train, cv_acc_test

def k_folds(dataset_size, k_folds):

    # -- Shuffle data indices
    fold_size = dataset_size // k_folds
    suffled_idx = np.arange(dataset_size)
    np.random.shuffle(suffled_idx)

    # -- Generator of shuffled indices for different folds
    for k in range(k_folds):
        test_idx = suffled_idx[k*fold_size:(k+1)*fold_size]
        train_idx = np.setdiff1d(suffled_idx, test_idx)
        # slower but no 'if' on either k = 0 for np.concatenate
        yield torch.tensor(train_idx), torch.tensor(test_idx), k


def compute_roc_auc_score(target, pred_labels):
    target = target.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    return roc_auc_score(target, pred_labels)



##############################################################################
# Functions for deeper GNNs                                                  
##############################################################################

# Missing feature vector setting, similar to PairNorm exps
def generate_missing_feature_setting(datadir, dataset, data, missing_rate=0):

    indices_dir = osp.join(datadir, dataset, 'indices')
    safe_mkdir(indices_dir)
    missing_indices_file = osp.join(indices_dir, "indices_missing_rate={}.npy".format(missing_rate))
    if not osp.exists(missing_indices_file):
        n = len(data.x)
        erasing_pool = torch.arange(n)[~data.train_mask] # keep training set always full feature
        size = int(len(erasing_pool) * (missing_rate/100))
        idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
        np.save(missing_indices_file, idx_erased)
    else:
        idx_erased = np.load(missing_indices_file)
      
    # erasing feature for random missing 
    if missing_rate > 0:
        data.x[idx_erased] = 0
    return data


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


def scale_neighbors(src, index, dim, dim_size, constant=1):
    mean = scatter(src = src, index = index, dim = dim, dim_size = dim_size, reduce = 'mean')
    red_src = (src - torch.index_select(mean, 0, index))**2
    std = torch.sqrt(scatter(src = red_src, index = index , dim = dim, dim_size = dim_size, reduce = 'add')+ 1e-16) # we add the 1e-16 for numerical instabilities (grad of sqrt at 0 gives nan)

    scaled_src = constant * src / torch.index_select(std, 0, index)
    scaled_src[torch.isinf(scaled_src)] = src[torch.isinf(scaled_src)]
    return scaled_src

# class NeighborNorm(nn.Module):

#     def __init__(self,constant=1):
#         super(NeighborNorm, self).__init__()
#         self.constant = constant

#     def forward(self, x, edge_index, dim=0):
#         mean = scatter(src = x, index = edge_index[0], dim = dim, dim_size = x.size(dim), reduce = 'mean')
#         reduced_x = (x - torch.index_select(mean, 0, edge_index[0]))**2
#         std = torch.sqrt(scatter(src = red_src, index = index , dim = dim, dim_size = dim_size, reduce = 'add')+ 1e-16) # we add the 1e-16 for numerical instabilities (grad of sqrt at 0 gives nan)

#         std = torch.sqrt(scatter(src = reduced_x, index = edge_index[0], dim = dim, dim_size= x.size(dim), reduce = 'add')+ 1e-16) # we add the 1e-16 for numerical instabilities (grad of sqrt at 0 gives nan)
#         out =  self.constant * x / torch.index_select(std, 0, edge_index[0])
#         infs = torch.isinf(out)
#         out[infs] = x[infs]
#         return out 
    
    
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n.replace('.weight','').replace('layers.',''))
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()