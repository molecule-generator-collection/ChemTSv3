from abc import ABC, abstractmethod
from typing import Self
from rdkit.Chem import Mol

class Node(ABC):
  def __init__(self, parent=None, lastprob=1.0):
    self.parent = parent
    self.children = {}
    self._probs = [] #for save, call probs()
    self.lastprob = lastprob #Prob(parent -> this node)
    self.n = 0 #visit count
    self.sum_r = 0.0 #sum of rewards
    self.mean_r = 0.0 #mean of rewards

  @abstractmethod
  def __str__(self) -> str:
    pass

  @abstractmethod
  def nextnodes(self) -> list[Self]:
    pass

  @abstractmethod
  def is_terminal(self) -> bool:
    pass
  
class MolConvertibleNode(Node):
  @abstractmethod
  def mol(self) -> Mol:
    pass