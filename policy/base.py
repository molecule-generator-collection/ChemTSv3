from abc import ABC, abstractmethod
from typing import Any
from node import Node

class Policy(ABC):
  @staticmethod
  @abstractmethod
  def evaluate(node: Node, conf: dict[str, Any]=None):
      pass