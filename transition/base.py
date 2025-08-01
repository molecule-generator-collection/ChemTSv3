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
    def next_nodes(self, node: Node) -> list[Node]:
        """Return the list of the child nodes. If the node is terminal, an empty list [] should be returned."""
        pass

    def rollout(self, initial_node: Node) -> Node:
        """Sample an offspring with has_reward() = True"""
        if initial_node.is_terminal():
            return initial_node
        
        current_node = initial_node
        while True:
            transitions = self.transitions(current_node)
            if not transitions:
                current_node.mark_as_terminal()
                return current_node
            
            _, next_nodes, probs = zip(*transitions)
            next_node = random.choices(next_nodes, weights=probs, k=1)[0]
            
            if next_node.has_reward():
                return next_node
            
            current_node = next_node
            
    def transitions(self, node: Node) -> list[tuple[Any, Node, float]]:
        """Return the list of (action, node, probability) tuples."""
        transitions = []
        for node in self.next_nodes(node):
            action = node.last_action
            prob = node.last_prob
            transitions.append((action, node, prob))
        return transitions
    
    # should be overridden if not inf
    def max_length(self) -> int:
        return 10**18
    
class LanguageModel(Transition):
    def __init__(self, lang: Language, logger: logging.Logger=None):
        self.lang = lang
        super().__init__(logger)

    @abstractmethod
    def next_nodes(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
        pass

    # override
    def max_length(self) -> int:
        return 10**18
    
class BlackBoxTransition(Transition):
    def __init__(self, n_samples=2, logger: logging.Logger=None):
        self.n_samples = n_samples
        super().__init__(logger)    

    @abstractmethod
    def sample_transition(self, node: Node) -> Node:
        """Sample one child node."""
        pass
    
    # implement
    def next_nodes(self, node):
        children = []
        for i in range(self.n_samples):
            next_node = self.sample_transition(node)
            next_node.last_action = i
            next_node.last_prob = 1 / self.n_samples
            children.append(next_node)
        return children