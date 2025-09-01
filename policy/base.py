from abc import ABC, abstractmethod
import random
from typing import Iterable
from node import Node

class Policy(ABC):
    @abstractmethod
    def select_child(self, node: Node) -> Node:
        """Select one child of the given node. Must not be called if node.children is empty."""
        pass
    
    def observe(self, child: Node, objective_values: list[float], reward: float):
        """Policies can update their internal state when observing the evaluation value of the node. By default, this method does nothing."""
        return
    
class TemplatePolicy(Policy):
    """
    Policy with progressive widening.
    Progressive widening ref: https://hal.science/hal-00542673v2/document
    """
    def __init__(self, pw_c: float=None, pw_alpha: float=None):
        if pw_c is None and pw_alpha is not None or pw_c is not None and pw_alpha is None:
            raise ValueError("Specify both (or none) of 'pw_c' and 'pw_alpha'.")
        
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha

    @abstractmethod
    def select_child(self, node: Node) -> Node:
        """Select one child of the given node. Must not be called if node.children is empty."""
        pass
    
    def candidates(self, node: Node) -> list[Node]:
        """Return reduced child candidates with progressive widening."""
        children = sorted(node.children, key=lambda c: (c.last_prob or 0.0), reverse=True)
        k = max(1, int(self.pw_c * (node.n ** self.pw_alpha))) if self.pw_c is not None else len(children)
        return children[:min(k, len(children))]

class ValuePolicy(TemplatePolicy):
    """Policy that selects the node with the highest score."""

    @abstractmethod
    def evaluate(self, node: Node) -> float:
        """Return the selection score of the given child node."""
        pass
    
    def select_child(self, node: Node) -> Node:
        candidates = self.candidates(node)
        max_y = max(self.evaluate(child) for child in candidates)
        best_candidates = [child for child in candidates if self.evaluate(child) == max_y]
        weights = [child.last_prob for child in best_candidates]
        return random.choices(best_candidates, weights=weights, k=1)[0]