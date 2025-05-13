from abc import ABC, abstractmethod
from language.language import Language
from node.node import Node
from node.sentencenode import SentenceNode

class EdgePredictor(ABC):
  def __init__(self, name=None):
    if name is not None:
      self.name = name
    else:
      self.name = "model"

  @abstractmethod
  def nextnodes_with_probs(self, node: Node) -> list[Node]:
    pass

  @abstractmethod
  def randomgen(self, initial_node: Node, conf: dict = None) -> Node:
    pass
  
class LanguageModel(EdgePredictor):
  def __init__(self, lang: Language, name=None):
    self.lang = lang
    super().__init__(name)

  #override
  @abstractmethod
  def nextnodes_with_probs(self, node: SentenceNode) -> list[SentenceNode]:
    pass

  #override
  @abstractmethod
  def randomgen(self, initial_node: SentenceNode, conf: dict = None) -> SentenceNode:
    pass

  @abstractmethod
  def max_length(self) -> int:
    pass