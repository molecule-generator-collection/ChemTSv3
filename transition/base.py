from abc import ABC, abstractmethod
import logging
from typing import Any
from language import Language
from node import Node, SentenceNode
from utils import select_indices_by_threshold

class Transition(ABC):
    def __init__(self, name=None, logger: logging.Logger=None):
        self.name = name or self.name or self.default_name()
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def transitions(self, node: Node) -> list[tuple[Any, Node]]:
        pass

    def default_name(self):
        return "No Name"

class WeightedTransition(Transition):
    @abstractmethod
    def _transitions_with_probs_impl(self, node: Node) -> list[tuple[Any, Node, float]]:
        pass

    # can implement default execution later
    @abstractmethod
    def rollout(self, initial_node: Node, **kwargs) -> Node:
        pass

    def transitions_with_probs(self, node: Node, threshold: float=None) -> list[tuple[Any, Node, float]]:
        if threshold is not None:
            raw_transitions = self._transitions_with_probs_impl(node)
            actions, nodes, probs = zip(*raw_transitions)
            remaining_indices = select_indices_by_threshold(probs, threshold)
            return [(actions[i], nodes[i], probs[i]) for i in remaining_indices]
        else:
            return self._transitions_with_probs_impl(node)

    # implement
    def transitions(self, node: Node) -> list[tuple[Any, Node]]:
        return self.transitions_with_probs(node)[:-1]
    
    # should override if not inf
    def max_length(self) -> int:
        return 10**18
    
class LanguageModel(WeightedTransition):
    def __init__(self, lang: Language, name=None, logger: logging.Logger=None):
        self.lang = lang
        super().__init__(name, logger)

    @abstractmethod
    def _transitions_with_probs_impl(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
        pass

    @abstractmethod
    def rollout(self, initial_node: SentenceNode, **kwargs) -> SentenceNode:
        pass

    # should override if not inf
    # override
    def max_length(self) -> int:
        return 10**18