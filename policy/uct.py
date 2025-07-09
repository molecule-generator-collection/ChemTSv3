from math import log, sqrt
from typing import Callable
from node import Node
from policy import ValuePolicy
from utils import PointCurve

class UCT(ValuePolicy):
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, best_rate: float=0.0, prior: float=None, prior_weight: int=1, max_prior: float=None):
        if type(c) == Callable:
            self.c = c
        elif type(c) == list:
            self.c = PointCurve(c)
        else:
            self.c = c
        self.best_ratio = best_rate
        self.prior = prior
        self.prior_weight = prior_weight
        self.max_prior = max_prior
        
    def get_prior(self, node: Node) -> tuple[float, int]:
        """Return prior value and prior weight."""
        if self.prior is not None:
            return self.prior, self.prior_weight
        else:
            return 0, 0

    def get_c_value(self, node: Node) -> float:
        if type(self.c) == Callable:
            c = self.c(node.depth)
        elif type(self.c) == PointCurve:
            c = self.c.curve(node.depth)
        else:
            c = self.c
        return c
    
    def get_exploration_term(self, node: Node):
        c = self.get_c_value(node)
        _, prior_weight = self.get_prior(node)
        n = node.n + prior_weight
        parent_n = node.parent.n + prior_weight
        return c * sqrt(log(parent_n) / (n))
    
    def get_mean_r(self, node: Node):
        prior, prior_weight = self.get_prior(node)
        sum_r = node.sum_r + prior * prior_weight
        n = node.n + prior_weight
        return sum_r / n

    # implement
    def evaluate(self, node: Node) -> float:
        if node.n == 0:
            return 10**9 # tiebreaker is implemented in policy base
        
        u = self.get_exploration_term(node)
        mean_r = self.get_mean_r(node)
        best_r = node.best_r
        if self.max_prior is not None:
            best_r = max(self.max_prior, best_r)
        return (1 - self.best_ratio) * mean_r + self.best_ratio * best_r + u