from abc import ABC, abstractmethod
from typing import Self, Any
from rdkit.Chem import Mol

class Node(ABC):
  def __init__(self, parent=None, last_action: Any=None, last_prob=1.0):
    self.parent = parent
    if parent is not None:
      self.depth = parent.depth + 1
    else:
      self.depth = 0
    self.children: dict[Any, Self] = {}
    self._probs = [] #for save, call probs()
    self.last_prob = last_prob #Prob(parent -> this node)
    self.last_action = last_action
    self.n = 0 #visit count
    self.sum_r = 0.0 #sum of rewards
    self.mean_r = 0.0 #mean of rewards

  @abstractmethod
  def __str__(self) -> str:
    pass

  @abstractmethod
  def is_terminal(self) -> bool:
    pass

  def add_child(self, action: Any, child: Self):
    self.children[action] = child
  
  def observe(self, value: float):
    self.n += 1
    self.sum_r += value
    self.mean_r = self.sum_r / self.n
  
class MolNode(Node):
  def __init__(self, parent=None, last_prob=1.0):
    self._is_valid_mol = None
    super().__init__(parent, last_prob)

  @abstractmethod
  def mol(self) -> Mol:
    #recommended to cache _is_valid_mol
    pass
  
  def is_valid_mol(self) -> bool:
    if self._is_valid_mol is not None:
      return self._is_valid_mol
    
    mol = self.mol()
    is_valid = not (mol is None or mol.GetNumAtoms()==0)
    self._is_valid_mol = is_valid
    return is_valid