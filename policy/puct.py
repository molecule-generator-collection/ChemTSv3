from math import sqrt
from node import Node
from policy import Policy
from typing import Callable
from utils import PointCurve

class PUCT(Policy):
    forced_rollout: bool # whether to return inf score for unexplored node or not
    initial_mean: float
    
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, initial_mean = 100):
        if type(c) == Callable:
            self.c = c
        elif type(c) == list:
            self.c = PointCurve(c)
        else:
            self.c = c
        self.initial_mean = initial_mean
        
    # implement
    def evaluate(self, node: Node):
        if type(self.c) == Callable:
            c = self.c(node.depth)
        elif type(self.c) == PointCurve:
            c = self.c.curve(node.depth)
        else:
            c = self.c

        u = c * node.last_prob * sqrt(node.parent.n) / (1 + node.n)
        if node.n == 0:
            return self.initial_mean + u
        return node.mean_r() + u