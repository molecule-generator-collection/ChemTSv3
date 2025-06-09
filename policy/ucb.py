from math import log, sqrt
from typing import Callable
from node import Node
from policy import Policy
from utils import make_curve_from_points

class UCB(Policy):
    c: Callable[[float], float] #depth -> exploration parameter
    
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, initial_mean = float("inf")):
        if type(c) == Callable:
            self.c = c
        elif type(c) == list:
            self.c = make_curve_from_points(c)
        else:
            self.c = lambda x: c
        self.initial_mean = initial_mean
    
    #implement
    def evaluate(self, node: Node):
        #c: exploration parameter
        if node.n == 0:
            return self.initial_mean + self.c(node.depth) * sqrt(2 * log(node.parent.n + 1) / (node.n + 1))
        u = self.c(node.depth) * sqrt(2 * log(node.parent.n) / (node.n))
        return node.mean_r + u