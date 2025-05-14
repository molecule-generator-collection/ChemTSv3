from abc import ABC, abstractmethod
from typing import Any
from language import Language
from node import Node, SentenceNode

class EdgePredictor(ABC):
  def __init__(self, name=None):
    if name is not None:
      self.name = name
    else:
      self.name = "model"

  @abstractmethod
  def child_candidates_with_probs(self, node: Node) -> list[Node]:
    pass

  @abstractmethod
  def generate(self, initial_node: Node, conf: dict[str, Any]=None) -> Node:
    pass
  
class LanguageModel(EdgePredictor):
  def __init__(self, lang: Language, name=None):
    self.lang = lang
    super().__init__(name)

  #override
  @abstractmethod
  def child_candidates_with_probs(self, node: SentenceNode) -> list[SentenceNode]:
    pass

  #override
  @abstractmethod
  def generate(self, initial_node: SentenceNode, conf: dict[str, Any]=None) -> SentenceNode:
    pass

  @abstractmethod
  def max_length(self) -> int:
    pass