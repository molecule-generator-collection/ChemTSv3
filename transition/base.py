from abc import ABC, abstractmethod
import logging
import random
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

    def rollout(self, initial_node: Node) -> Node:
        """samples an offspring with has_reward() = True"""
        if initial_node.is_terminal():
            return initial_node
        
        current_node = initial_node
        while True:
            transitions = self.transitions_with_probs(current_node)
            if not transitions:
                current_node.mark_as_terminal()
                return current_node
            
            _, next_nodes, probs = zip(*transitions)
            next_node = random.choices(next_nodes, weights=probs, k=1)[0]
            
            if next_node.has_reward():
                return next_node
            
            current_node = next_node

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

    # override
    def max_length(self) -> int:
        return 10**18