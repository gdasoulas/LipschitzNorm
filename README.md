# Repository for ICML 2021 paper: [Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks](https://arxiv.org/abs/2103.04886). 
This repo contains the normalization implementation for the paper "Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks."

### Dependencies
Before running the experiments, check the dependent python packages. 

The list of the dependencies can be found in the `requirements.txt` file at the root of the project.

### Models & Normalizations
Under the  subdirectory `models/`, the models of DeepGAT and DeepGT are located. Under the subdirectory `normalizations/`, the proposed normalizations
LipscitzNorm (for the GAT layer) and the Quadratic_LipschitzNorm (for the Graph Transformer layer) are located.

Specifically, the proposed normalizations are located in the modules:

`lipschitznorm.py` and `quadratic_lipschitznorm.py`.

### Data
The data are automatically downloaded and processed with the first run of a training module, as shown in the next paragraph.

### Experimentation files
The main files for experimentation are at the root of the project:
- `train_deepgat_missing.py`: This file contains the experiment of node classification task under the missing vector setting using GAT model.
- `train_deepgat_depth.py`: This file contains the experiment of node classification task using GAT model with respect to the increasing depth.
- `train_deepgt_depth.py`: This file contains the experiment of node classification task using Graph Transformer model with respect to the increasing depth.

### Examples of training
- For the missing vector setting, an example of an experiment is 

`python train_deepgat_missing.py --dataset Cora --hidden_dim 64 --norm lipschitznorm --num_layers 5 --heads 4 --nepoch 1000 --lr 0.005`

- For the model depth setting, an example of an experiment is 

`python train_deepgat_depth.py --dataset Cora --hidden_dim 64 --heads 8 --nepoch 1000 --lr 0.005`

