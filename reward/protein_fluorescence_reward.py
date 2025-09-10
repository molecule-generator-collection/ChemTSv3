import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import itertools
import pandas as pd
from net.rew import BaseCNN
from utils.eval_utils import distance
from utils.constants import seq_to_one_hot
from node import FASTAStringNode, SentenceNode
from reward import Reward

class ProteinFluorescenceReward(Reward):
    """
    Fetched from: https://github.com/haewonc/LatProtRL/blob/main/metric.py
    """
    def __init__(self):
        pass

    # implement
    def objective_functions(self):
        def fitness(node: FASTAStringNode | SentenceNode):
            if issubclass(node.__class__, FASTAStringNode):
                fasta = node.string
            else:
                fasta = node.key()
            return 0

        return [fitness]
    
    def reward_from_objective_values(self, objective_values) -> float:
        pass
    
class OnehotDataset(Dataset):
  def __init__(self, seqs):
    self.seqs = seqs 
  def __len__(self):
    return len(self.seqs)
  def __getitem__(self, index):
    return seq_to_one_hot(self.seqs[index])

class Evaluator:
  def __init__(self, protein, max_target, min_target, device, batch_size = 16):
    self.device = device 
    self.batch_size = batch_size
    self.max_target, self.min_target = max_target, min_target
    oracle = BaseCNN(make_one_hot=False)
    oracle_ckpt = torch.load(f'ckpt/{protein}/oracle.ckpt', map_location=self.device)
    if "state_dict" in oracle_ckpt.keys():
        oracle_ckpt = oracle_ckpt["state_dict"]
    oracle.load_state_dict({ k.replace('predictor.',''):v for k,v in oracle_ckpt.items() })
    oracle.eval()
    self.oracle = oracle.to(device)
    high = pd.read_csv(f'data/{protein}/all.csv')[['sequence', 'target']]
    high = high[high['target'] > high['target'].quantile(q=0.9).item()]
    self.high = high['sequence'].tolist()
    self.high = self.high[:128]
    
  def evaluate(self, seqs, inits):
    dataset = OnehotDataset(seqs)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    targets = []
    with torch.no_grad():
      for batch in dataloader:
        _, target = self.oracle(batch.to(self.device), get_embed=True)
        target = (target - self.min_target) / (self.max_target - self.min_target)
        targets.extend(list(target.cpu().flatten()))
    fitness = np.median(targets)
    
    distances = []
    for s1, s2 in itertools.combinations(seqs, 2):
        distances.append(distance(s1, s2))
    diversity = np.median(distances)
    
    distances = []
    for j in seqs:
        dist_j = []
        for i in inits:
            dist_j.append(distance(i,j))
        distances.append(min(dist_j))
    novelty = np.median(distances)
    
    distances = []
    for j in seqs:
        dist_j = []
        for i in self.high:
            dist_j.append(distance(i,j))
        distances.append(min(dist_j))
    high = np.median(distances)


    return fitness, diversity, novelty, high