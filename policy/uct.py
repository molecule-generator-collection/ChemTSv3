from math import log, sqrt
from typing import Callable
from node import Node
from policy import ValuePolicy
from utils import PointCurve

class UCT(ValuePolicy):
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=0.3, best_rate: float=0.0, max_prior: float=None, pw_c: float=None, pw_alpha: float=None):
        """
        Args:
            c: The weight of the exploration term. Higher values place more emphasis on exploration over exploitation.
            best_rate: A value between 0 and 1. The exploitation term is computed as 
                       best_rate * (best reward) + (1 - best_rate) * (average reward).
            max_prior: A lower bound for the best reward. If the actual best reward is lower than this value, this value is used instead.
            pw_c: Used for progressive widening.
            pw_alpha: Used for progressive widening.
        """
        if type(c) == Callable:
            self.c = c
        elif type(c) == list:
            self.c = PointCurve(c)
        else:
            self.c = c
        self.best_ratio = best_rate
        self.max_prior = max_prior
        super().__init__(pw_c=pw_c, pw_alpha=pw_alpha)

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
        return c * sqrt(log(node.parent.n) / (node.n))

    # implement
    def evaluate(self, node: Node) -> float:
        if node.n == 0:
            return 10**9 # tiebreaker is implemented in policy base
        
        mean_r = node.sum_r / node.n
        u = self.get_exploration_term(node)
        best_r = node.best_r
        if self.max_prior is not None:
            best_r = max(self.max_prior, best_r)
        return (1 - self.best_ratio) * mean_r + self.best_ratio * best_r + u