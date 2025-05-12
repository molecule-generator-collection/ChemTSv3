from abc import abstractmethod
from language.language import Language
from .edgepredictor import EdgePredictor
from node.sentencenode import SentenceNode

class LanguageModel(EdgePredictor):
  def __init__(self, lang: Language, name=None):
    self.lang = lang
    super().__init__(name)

  #override
  @abstractmethod
  def nextnodes_with_probs(self, node: SentenceNode) -> list[SentenceNode]:
    raise NotImplementedError("edgeprobs() needs to be implemented.")

  #override
  @abstractmethod
  def randomgen(self, initial_node: SentenceNode) -> SentenceNode:
    raise NotImplementedError("randomgen() needs to be implemented.")

  @abstractmethod
  def max_length(self) -> int:
    raise NotImplementedError("max_length() needs to be implemented.")
