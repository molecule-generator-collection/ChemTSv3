from abc import ABC, abstractmethod
import logging
from typing import Any
from language import Language
from node import Node, SentenceNode

class Transition(ABC):
    def __init__(self, logger: logging.Logger=None):
        self.logger = logger or logging.getLogger(__name__)    

    @abstractmethod
    def transitions_with_probs(self, node: Node) -> list[tuple[Any, Node, float]]:
        """returns the list of (action, node, probability)"""
        pass
    
    # TODO: implement default execution
    # should return the initial_node itself if it's terminal
    @abstractmethod
    def rollout(self, initial_node: Node, **kwargs) -> Node:
        """randomly samples a node with has_reward() = True"""
        pass

    def transitions(self, node: Node) -> list[tuple[Any, Node]]:
        return self.transitions_with_probs(node)[:-1]
    
    # should be overridden if not inf
    def max_length(self) -> int:
        return 10**18
    
class LanguageModel(Transition):
    def __init__(self, lang: Language, logger: logging.Logger=None):
        self.lang = lang
        super().__init__(logger)

    @abstractmethod
    def transitions_with_probs(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
        pass

    @abstractmethod
    def rollout(self, initial_node: SentenceNode, **kwargs) -> SentenceNode:
        pass

    # override
    def max_length(self) -> int:
        return 10**18