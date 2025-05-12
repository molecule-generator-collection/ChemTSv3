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
    raise NotImplementedError("__str__() needs to be implemented.")

  @abstractmethod
  def nextnodes(self) -> list[Self]:
    raise NotImplementedError("nextnodes() needs to be implemented.")

  @abstractmethod
  def is_terminal(self) -> bool:
    raise NotImplementedError("is_terminal() needs to be implemented.")
  
class MolConvertibleNode(Node):
  @abstractmethod
  def mol(self) -> Mol:
    raise NotImplementedError("mol() needs to be implemented.")