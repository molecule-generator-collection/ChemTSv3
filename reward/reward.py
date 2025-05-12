from abc import ABC, abstractmethod
from typing import List, Callable
from node.node import Node, MolConvertibleNode
from rdkit.Chem import Mol

class Reward(ABC):
  #Callable[[Node], float] instead of Callable[[Mol], float] for better compatibility
  @staticmethod
  @abstractmethod
  def objective_functions(conf: dict) -> List[Callable[[Node], float]]:
    pass

  @staticmethod
  @abstractmethod
  def reward_from_objective_values(values: List[float], conf: dict) -> float:
    pass

  @classmethod
  def objective_values(cls, node: Node, conf: dict):
    return [f(node) for f in cls.objective_functions(conf=conf)]
  
  @classmethod
  def objective_values_and_reward(cls, node: Node, conf: dict) -> tuple[list[float], float]:
    objective_values = cls.objective_values(node, conf=conf)
    reward = cls.reward_from_objective_values(objective_values, conf=conf)
    return objective_values, reward

class MolReward(Reward):
  @staticmethod
  @abstractmethod
  def mol_objective_functions(conf: dict) -> List[Callable[[Mol], float]]:
    pass

  @staticmethod
  @abstractmethod
  def reward_from_objective_values(values: List[float], conf: dict) -> float:
    pass

  #override
  @classmethod
  def objective_functions(cls, conf: dict) -> List[Callable[[MolConvertibleNode], float]]:
    return [lambda node: f(node.mol()) for f in cls.mol_objective_functions(conf)]