from abc import ABC, abstractmethod
from typing import Any, List, Callable
from node import Node, MolNode
from rdkit.Chem import Mol

class Reward(ABC):
  #Callable[[Node], float] instead of Callable[[Mol], float] for better compatibility
  @classmethod
  @abstractmethod
  def objective_functions(cls, **kwargs) -> List[Callable[[Node], float]]:
    pass

  @staticmethod
  @abstractmethod
  def reward_from_objective_values(values: List[float], **kwargs) -> float:
    pass

  @classmethod
  def objective_values(cls, node: Node, **kwargs):
    return [f(node) for f in cls.objective_functions(**kwargs)]
  
  @classmethod
  def objective_values_and_reward(cls, node: Node, objective_values_conf: dict[str, Any], reward_conf: dict[str, Any]) -> tuple[list[float], float]:
    objective_values = cls.objective_values(node, **objective_values_conf)
    reward = cls.reward_from_objective_values(objective_values, **reward_conf)
    return objective_values, reward

class MolReward(Reward):
  @staticmethod
  @abstractmethod
  def mol_objective_functions(**kwargs) -> List[Callable[[Mol], float]]:
    pass

  @staticmethod
  @abstractmethod
  def reward_from_objective_values(values: List[float], **kwargs) -> float:
    pass

  @staticmethod
  def wrap_with_mol(f):
    def wrapper(node: Node):
      return f(node.mol())
    wrapper.__name__ = f.__name__ #copy function names
    return wrapper

  #override
  @classmethod
  def objective_functions(cls, **kwargs) -> List[Callable[[MolNode], float]]:
    return [MolReward.wrap_with_mol(f) for f in cls.mol_objective_functions(**kwargs)]