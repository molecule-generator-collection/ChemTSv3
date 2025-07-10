from math import log, sqrt
from typing import Callable
from node import Node
from policy import ValuePolicy
from utils import PointCurve

class UCT(ValuePolicy):
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, best_rate: float=0.0, prior: float=None, prior_weight: int=None, force_first_visit: bool=True, max_prior: float=None):
        if prior_weight is not None and prior_weight < 0:
            raise ValueError("'prior_weight' must be >= 0.")        

        if type(c) == Callable:
            self.c = c
        elif type(c) == list:
            self.c = PointCurve(c)
        else:
            self.c = c
        self.best_ratio = best_rate
        self.prior = prior
        if prior_weight is not None:
            self.prior_weight = prior_weight
        elif prior is not None:
            self.prior_weight = 1
        else:
            self.prior_weight = 0
        self.max_prior = max_prior
        self.force_first_visit = force_first_visit
        
    def get_prior(self, node: Node) -> float:
        """Return prior value (None if not using prior)."""
        return self.prior

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
        n = max(node.n + self.prior_weight, 1) # can be 0+0 with prior value (use prior only for n=0)
        parent_n = max(node.parent.n + self.prior_weight, 1)
        return c * sqrt(log(parent_n) / (n))
    
    def get_mean_r(self, node: Node):
        prior = self.get_prior(node)
        if prior is None and node.n == 0:
            return None

        if prior is None:
            sum_r = node.sum_r
        elif self.prior_weight == 0 and node.n == 0:
            sum_r = prior
        else:
            sum_r = node.sum_r + prior * self.prior_weight
                
        n = max(node.n + self.prior_weight, 1)
        return sum_r / n

    # implement
    def evaluate(self, node: Node) -> float:
        mean_r = self.get_mean_r(node)
        
        if mean_r == None or node.n == 0 and self.force_first_visit:
            return 10**9 # tiebreaker is implemented in policy base
        
        u = self.get_exploration_term(node)
        best_r = node.best_r
        if self.max_prior is not None:
            best_r = max(self.max_prior, best_r)
        return (1 - self.best_ratio) * mean_r + self.best_ratio * best_r + u