from math import log, sqrt
from typing import Callable
from node import Node
from policy import Policy
from utils import PointCurve

# not named "UCT" and has 2* before log in favor of ChemTSv2 compatibility 
class UCB(Policy):
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, initial_mean = 10**9, best_rate: float=0.0):
        if type(c) == Callable:
            self.c = c
        elif type(c) == list:
            self.c = PointCurve(c)
        else:
            self.c = c
        self.initial_mean = initial_mean
        self.best_ratio = best_rate
    
    # implement
    def evaluate(self, node: Node):
        if type(self.c) == Callable:
            c = self.c(node.depth)
        elif type(self.c) == PointCurve:
            c = self.c.curve(node.depth)
        else:
            c = self.c

        if node.n == 0:
            return self.initial_mean + c * sqrt(2 * log(node.parent.n + 1) / (node.n + 1)) 
        u = c * sqrt(2 * log(node.parent.n) / (node.n))
        return (1 - self.best_ratio) * node.mean_r() + self.best_ratio * node.best_r + u