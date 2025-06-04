from math import sqrt
from node import Node
from policy import Policy
from typing import Callable
from utils import make_curve_from_points

class PUCT(Policy):
    c: Callable[[float], float] #depth -> exploration parameter
    forced_rollout: bool #whether to return inf score for unexplored node or not
    
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, forced_rollout: bool=True):
        if type(c) == Callable:
            self.c = c
        elif type(c) == list:
            self.c = make_curve_from_points(c)
        else:
            self.c = lambda x: c
        self.forced_rollout = forced_rollout
        
    #implement
    def evaluate(self, node: Node):
        if node.n == 0 and self.forced_rollout:
            return float("inf")
        u = self.c(node.depth) * node.last_prob * sqrt(node.parent.n) / (1 + node.n)
        return node.mean_r + u