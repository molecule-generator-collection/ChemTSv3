from math import sqrt
from node import Node
from policy import Policy
from typing import Callable
from utils import make_curve_from_points

class PUCT(Policy):
    c: Callable[[float], float] #depth -> exploration parameter
    forced_rollout: bool #whether to return inf score for unexplored node or not
    initial_mean: float
    
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
        u = self.c(node.depth) * node.last_prob * sqrt(node.parent.n) / (1 + node.n)
        if node.n == 0:
            return self.initial_mean + u
        return node.mean_r + u