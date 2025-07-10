from abc import ABC, abstractmethod
import random
from typing import Any
from node import Node

class Policy(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select_child(self, node: Node) -> Node:
        pass
    
    def observe(self, child: Node, objective_values: list[float], reward: float):
        return
    
class ValuePolicy(Policy):
    """Policy that selects the node with the highest score"""
    @abstractmethod
    def evaluate(self, node: Node, **kwargs) -> float:
        pass
    
    def select_child(self, node: Node) -> Node:
        max_y = max(self.evaluate(child) for child in node.children)
        candidates = [child for child in node.children if self.evaluate(child) == max_y]
        weights = [child.last_prob for child in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]