from abc import ABC, abstractmethod
from typing import List, Callable
from node.node import Node

class Reward(ABC):
  #Callable[[Node], float] instead of Callable[[Mol], float] for better compatibility
  @staticmethod
  @abstractmethod
  def objective_functions(conf: dict) -> List[Callable[[Node], float]]:
    raise NotImplementedError("get_objective_functions() needs to be implemented")

  @staticmethod
  @abstractmethod
  def reward_from_objective_values(values: List[float], conf: dict) -> float:
    raise NotImplementedError("reward_from_objective_values() needs to be implemented")

  @classmethod
  def objective_values(cls, node: Node, conf: dict):
    return [f(node) for f in cls.get_objective_functions(conf=conf)]
  
  @classmethod
  def objective_values_and_reward(cls, node: Node, conf: dict) -> tuple[list[float], float]:
    objective_values = cls.objective_values(node, conf=conf)
    reward = cls.reward_from_objective_values(objective_values, conf=conf)
    return objective_values, reward