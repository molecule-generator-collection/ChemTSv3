import itertools
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from node import FASTAStringNode, SentenceNode
from reward import SingleReward

ORACLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/fluorescence/gfp_oracle.ckpt"))

ALPHABET = list("ARNDCQEGHILKMFPSTWYV")
IDXTOAA = {i: ALPHABET[i] for i in range(20)}
AATOIDX = {v: k for k, v in IDXTOAA.items()}

class ProteinFluorescenceReward(SingleReward):
    """
    Fetched from: https://github.com/haewonc/LatProtRL/blob/main/metric.py
    """
    def __init__(self, max_fitness=1.283419251, min_fitness=4.123108864, device: str=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.evaluator = Evaluator(max_target=max_fitness, min_target=min_fitness, device=self.device, batch_size=1)

    def reward(self, node: FASTAStringNode | SentenceNode) -> float:
        if issubclass(node.__class__, FASTAStringNode):
            fasta = node.string
        else:
            fasta = node.key()
            
        fitness = self.evaluator.evaluate([fasta])
        return fitness
    
class OnehotDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        return self.seq_to_one_hot(self.seqs[index])
    
    @staticmethod
    def seq_to_one_hot(seq):
        return F.one_hot(torch.tensor([AATOIDX[aa] for aa in list(seq)]), 20)

class Evaluator:
    def __init__(self, max_target, min_target, device, batch_size=1):
        self.device = device 
        self.batch_size = batch_size
        self.max_target, self.min_target = max_target, min_target
        oracle = BaseCNN(make_one_hot=False)
        oracle_ckpt = torch.load(ORACLE_PATH, map_location=self.device)
        if "state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["state_dict"]
        oracle.load_state_dict({ k.replace('predictor.',''):v for k,v in oracle_ckpt.items() })
        oracle.eval()
        self.oracle = oracle.to(device)
    
    def evaluate(self, seqs):
        dataset = OnehotDataset(seqs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        targets = []
        with torch.no_grad():
            for batch in dataloader:
                _, target = self.oracle(batch.to(self.device), get_embed=True)
                target = (target - self.min_target) / (self.max_target - self.min_target)
                targets.extend(list(target.cpu().flatten()))
            fitness = np.median(targets)

        return fitness
    
    @staticmethod
    def distance(s1, s2):
        return sum([1 if i!=j else 0 for i,j in zip(list(s1), list(s2))])

class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

        if activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(100.0*x)
        elif activation == 'softplus':
            self.act_fn = nn.Softplus()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.act_fn = lambda x: F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x

class BaseCNN(nn.Module):
    def __init__(
            self,
            n_tokens: int = 20,
            kernel_size: int = 5 ,
            input_size: int = 256,
            dropout: float = 0.0,
            make_one_hot=True,
            activation: str = 'relu',
            linear: bool=True,
            **kwargs):
        super(BaseCNN, self).__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size*2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout) # TODO: actually add this to model
        self.input_size = input_size
        self._make_one_hot = make_one_hot
    
    def get_embed(self, x):
        #onehotize
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        # encoder
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        # embed
        x = self.embedding(x)
        return x

    def forward(self, x, get_embed=False):
        #onehotize
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        # encoder
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        # embed
        x = self.embedding(x)
        # decoder
        output = self.decoder(x).squeeze(1)
        if get_embed:
            return x, output
        return output