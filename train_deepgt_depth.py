"""
Code for training Deep Graph Transformer on node classification tasks
Cora and PubMed with increasing depth
"""

from models.deepgt import DeepGT
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch_geometric.datasets import Planetoid
import tqdm, time
from pathlib import Path
from utils import *
import parser
import numpy as np

#########################################################
# Train/Test functions
#########################################################

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss_train = criterion(output[data.train_mask], data.y[data.train_mask])
    acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(data)
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])

    if opt.lr_patience > 0:
        print('test')
        lr_scheduler.step(loss_val)

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t)
          )

    whole_state = {
        'epoch': epoch,
        'loss_train': loss_train.item(),
        'loss_val': loss_val.item(),
        'acc_train': acc_train.item(),
        'acc_val': acc_val.item(),
        }        
    return whole_state


def test():
    model.eval()
    output = model(data)
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    print(f"Test accuracy= {acc_test.item()}")
    return acc_test.item()

# Parse arguments
opt = parser.parser() 

# Logs 
expfolder = None
if opt.save:
    now = time.strftime("%Y%m%d_%H%M%S")
    expfolder = os.path.join('./experiments', opt.dataset, now)
    Path(expfolder).mkdir(parents=True, exist_ok=True)

    with open(osp.join(expfolder, 'config.json'), 'w') as f:
        json.dump(opt.__dict__, f, sort_keys=True, indent=4)

if opt.device > -1:
    device = torch.device('cuda:'+str(opt.device))
else:
    device = torch.device('cpu')

transform = None
# we use the same train/val/test splits as Kipf & Welling (2016)
dataset = Planetoid(osp.join(opt.datadir, opt.dataset), opt.dataset, transform = transform)
data = dataset[0].to(device)
print(data)


for norm in ['None', 'layernorm','lipschitznorm','lipschitznorm+layernorm']:
    for num_layers in [1,5,10,15,20, 25, 30, 35]:
        accs = []
        for exp in range(opt.nexps):
            print('*******************************')
            print(f'Beginning experiment {exp} with normalization {norm} ...')
            print('*******************************')

            # model, optimizer and lr scheduler setup
            model = DeepGT(
                idim = dataset.num_features, 
                hdim = opt.hidden_dim,
                odim = dataset.num_classes,
                num_layers = num_layers,
                heads = opt.heads,
                norm = norm,
                beta =opt.beta
                ).to(device)
                        

            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0005)
            criterion = torch.nn.CrossEntropyLoss()
            print(model)
            # scheduler for learning rate decay
            lr_scheduler = None
            if opt.lr_patience > 0:
                # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_patience, gamma=0.9)
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

            # early stopping init
            if opt.es_patience > 0:
                es_patience = 400    # max number of epochs with no improvement 
                es_no_improve = 0
                min_val_loss = np.Inf

            epochs = tqdm.trange(opt.nepoch) 
            t_initial = time.time()
            lr = opt.lr
            states = []
            for epoch in epochs:
                state = train(epoch)   
                states.append(state)

                # early stopping 
                if opt.es_patience > 0:
                    if state['loss_val'] < min_val_loss:
                        es_no_improve = 0
                        min_val_loss = state['loss_val']
                    else:
                        es_no_improve += 1

                    if epoch > 5 and es_no_improve == es_patience:
                        print('Early Stopping !')
                        break


            print('Done!')
            if opt.save:
                if norm==None:
                    norm="None"
                torch.save(states, osp.join(expfolder,f'model={model.__class__.__name__}_norm={norm}_nl={num_layers}_nheads={opt.heads}_hdim={opt.hidden_dim}.pth'))

            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_initial))

            # Testing
            accs.append(test())
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f'Mean test accuracy: {mean_acc}')
