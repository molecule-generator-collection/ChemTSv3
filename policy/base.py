from abc import ABC, abstractmethod
import math
import random
from typing import Iterable
import numpy as np
from node import Node

class Policy(ABC):
    @abstractmethod
    def select_child(self, node: Node) -> Node:
        """Select one child of the given node. Must not be called if node.children is empty."""
        pass
    
    def observe(self, child: Node, objective_values: list[float], reward: float):
        """Policies can update their internal state when observing the evaluation value of the node. By default, this method does nothing."""
        return
    
    def candidates(self, node: Node) -> list[Node]:
        """Return available child candidates. Override this for progressive widening etc."""
        return node.children
    
    def sample_candidates(self, node: Node, max_size: int=1, replace: bool=False) -> list[Node]:
        cands = self.candidates(node)
        if not cands:
            return None
        size = min(max_size, len(cands))
        weights = np.array([node.last_prob for node in cands], dtype=np.float64)
        total = weights.sum()
        probabilities = weights / total
        return np.random.choice(cands, size=size, replace=replace, p=probabilities)
    
class TemplatePolicy(Policy):
    """
    Policy with progressive widening.
    Progressive widening ref: https://www.researchgate.net/publication/23751563_Progressive_Strategies_for_Monte-Carlo_Tree_Search
    """
    def __init__(self, pw_c: float=None, pw_alpha: float=None, pw_beta: float=0):
        if pw_c is None and pw_alpha is not None or pw_c is not None and pw_alpha is None:
            raise ValueError("Specify both (or none) of 'pw_c' and 'pw_alpha'.")
        
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha
        self.pw_beta = pw_beta

    @abstractmethod
    def select_child(self, node: Node) -> Node:
        """Select one child of the given node. Must not be called if node.children is empty."""
        pass
    
    def candidates(self, node: Node) -> list[Node]:
        """Return reduced child candidates with progressive widening."""
        children = sorted(node.children, key=lambda c: (c.last_prob or 0.0), reverse=True) # deterministic
        k = max(1, int(self.pw_c * (node.n ** self.pw_alpha) + self.pw_beta)) if self.pw_c is not None else len(children)
        return children[:min(k, len(children))]

class ValuePolicy(TemplatePolicy):
    """Policy that selects the node with the highest score."""

    @abstractmethod
    def evaluate(self, node: Node) -> float:
        """Return the selection score of the given child node."""
        pass
    
    def select_child(self, node: Node) -> Node:
        evals = []
        candidates = self.candidates(node)
        
        for c in candidates:
            try:
                y = self.evaluate(c)
            except Exception:
                y = float("-inf")
            if not math.isfinite(y):
                y = float("-inf")
            evals.append((c, y))
            
        if all(y == float("-inf") for _, y in evals):
                return random.choice(candidates)
        
        max_y = max(y for _, y in evals)
        eps = 1e-12
        best_candidates = [c for c, y in evals if y >= max_y - eps]
        if not best_candidates:
            best_candidates = [candidates[0]]

        # sample if tiebreaker
        weights = []
        for c in best_candidates:
            w = c.last_prob or 0
            weights.append(w)

        if sum(weights) <= 0:
            return random.choice(best_candidates)

        return random.choices(best_candidates, weights=weights, k=1)[0]