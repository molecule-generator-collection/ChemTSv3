from abc import ABC, abstractmethod
import random
from typing import Any
from node import Node

class Policy(ABC):
    @abstractmethod
    def select_child(self, node: Node) -> Node:
        """Select one child of the given node. Must not be called if node.children is empty."""
        pass
    
    def observe(self, child: Node, objective_values: list[float], reward: float):
        return
    
class ValuePolicy(Policy):
    """Policy that selects the node with the highest score"""
    @abstractmethod
    def evaluate(self, node: Node) -> float:
        """Return the selection score of the given child node."""
        pass
    
    def select_child(self, node: Node) -> Node:
        max_y = max(self.evaluate(child) for child in node.children)
        candidates = [child for child in node.children if self.evaluate(child) == max_y]
        weights = [child.last_prob for child in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]