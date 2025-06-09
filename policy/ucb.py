from math import log, sqrt
from typing import Callable
from node import Node
from policy import Policy
from utils import PointCurve

class UCB(Policy):
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, initial_mean = float("inf")):
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

        if node.n == 0:
            return self.initial_mean + c * sqrt(2 * log(node.parent.n + 1) / (node.n + 1))
        u = c * sqrt(2 * log(node.parent.n) / (node.n))
        return node.mean_r + u