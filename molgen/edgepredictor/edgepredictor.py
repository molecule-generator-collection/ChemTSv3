from abc import ABC, abstractmethod
from node.node import Node

class EdgePredictor(ABC):
  def __init__(self, name=None):
    if name is not None:
      self.name = name
    else:
      self.name = "model"

  @abstractmethod
  def nextnodes_with_probs(self, node: Node) -> list[Node]:
    raise NotImplementedError("edgeprobs() needs to be implemented.")

  @abstractmethod
  def randomgen(self, initial_node: Node) -> Node:
    raise NotImplementedError("randomgen() needs to be implemented.")