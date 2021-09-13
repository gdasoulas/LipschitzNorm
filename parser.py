import argparse

def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=0, help='-1: cpu; > -1: cuda device id')
    p.add_argument('--datadir', type=str, help='path to dataset', default='./data')
    p.add_argument('--dataset', type=str, default='Cora', help='dataset name for node classification, e.g Cora, CiteSeer')
    p.add_argument('--missing_rate', type=int, help='rate for missing feature vectors of unlabeled data', default='0')
    p.add_argument('--norm', type=str, default=None, help='Type of normalization.')
    p.add_argument('--beta', default=False, action='store_true', help='Type of normalization.')
    p.add_argument('--hidden_dim', type=int, help='Dimension of hidden layers', default=64)
    p.add_argument('--heads', type=int, help='Number of attention heads', default=8)
    p.add_argument('--nexps', type=int, default=5, help='Number of experiments')

    p.add_argument('--num_layers', type=int, help='number of iterations over the neighborhood', default=3)
    p.add_argument('--batch_size', type=int, default=16, help='batch size')

    p.add_argument('--train_ratio', type=float, help='only if fixed_splits = False, ratio of train split', )
    p.add_argument('--es_patience', type=int, default=-1, help='patience for early stopping criterion')

    p.add_argument('--lr', type=float, help='learning rate', default=0.01)
    p.add_argument('--lr_patience', type=int, default=50, help='number of epoch to wait before trigerring lr decay')
    p.add_argument('--lr_min', type=float, help='learning rate threshold where to stop the training', default=0)
    p.add_argument('--weight_decay', type=float, help='L2 penalty (default=0)', default=0.0005)

    p.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    p.add_argument('--verbose', type=int, help='Verbosity level, 0 means not verbose', default=0)
    p.add_argument('--outputdir', type=str, help='path to save xp', default='experiments')
    p.add_argument('--xp', type=str, help='xp name', default='single_exp')
    p.add_argument('--save',  action='store_true', default=False)

    opt = p.parse_args()
    return opt