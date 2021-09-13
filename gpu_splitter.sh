#!/bin/bash
screen -dmS maxnorm bash -c 'python train_deepgat_cora.py --device 0 --hidden_dim 64 --dataset Cora --nepoch 400 --save' 
screen -dmS maxnorm bash -c 'python train_deepgt_cora.py --device 1 --hidden_dim 64 --dataset Cora --nepoch 400 --lr 0.001 --save'
