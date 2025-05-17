from abc import ABC, abstractmethod
from typing import Any
import logging
from language import Language
from node import Node, SentenceNode

class Transition(ABC):
  def __init__(self, name=None, logger: logging.Logger=None):
    self.name = name or self.default_name()
    self.logger = logger or logging.getLogger(__name__)

  @abstractmethod
  def transitions(self, node: Node) -> list[tuple[Any, Node]]:
    pass

  def default_name(self):
    return "No Name"

class WeightedTransition(Transition):
  @abstractmethod
  def transitions_with_probs(self, node: Node) -> list[tuple[Any, Node, float]]:
    pass

  #can implement default execution later
  @abstractmethod
  def rollout(self, initial_node: Node, **kwargs) -> Node:
    pass

  #override
  def transitions(self, node: Node) -> list[tuple[Any, Node]]:
    return self.transitions_with_probs(node)[:-1]
  
  #should override if not inf
  def max_length(self) -> int:
    return 10**18
  
class LanguageModel(WeightedTransition):
  def __init__(self, lang: Language, name=None, logger: logging.Logger=None):
    self.lang = lang
    super().__init__(name, logger)

  #override
  @abstractmethod
  def transitions_with_probs(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
    pass

  #override
  @abstractmethod
  def rollout(self, initial_node: SentenceNode, **kwargs) -> SentenceNode:
    pass

  #should override if not inf
  #override
  def max_length(self) -> int:
    return 10**18