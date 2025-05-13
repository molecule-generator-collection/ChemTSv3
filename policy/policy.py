from abc import ABC, abstractmethod
from node import Node

class Policy(ABC):
  @staticmethod
  @abstractmethod
  def evaluate(node: Node, conf: dict = None):
      pass