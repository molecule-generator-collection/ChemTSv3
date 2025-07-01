from math import sqrt
from node import Node
from policy import ValuePolicy
from typing import Callable
from utils import PointCurve

class PUCT(ValuePolicy):
    forced_rollout: bool # whether to return inf score for unexplored node or not
    initial_mean: float
    
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
        
    # implement
    def evaluate(self, node: Node) -> float:
        if type(self.c) == Callable:
            c = self.c(node.depth)
        elif type(self.c) == PointCurve:
            c = self.c.curve(node.depth)
        else:
            c = self.c
            
        n = node.n
        parent_n = node.parent.n
        sum_r = node.sum_r
        if self.prior is not None:
            n += self.prior_weight
            parent_n += self.prior_weight
            sum_r += self.prior * self.prior_weight

        if node.n == 0:
            return 10**9 + node.last_prob

        u = c * node.last_prob * sqrt(parent_n) / (1 + n)
        mean_r = sum_r / n
        best_r = node.best_r
        if self.max_prior is not None:
            best_r = max(self.max_prior, best_r)
        return (1 - self.best_ratio) * mean_r + self.best_ratio * best_r + u