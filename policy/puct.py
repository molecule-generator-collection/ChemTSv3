from math import sqrt
from node import Node
from policy import Policy

class PUCT(Policy):
    def __init__(self, c: float=1, forced_rollout: bool=True):
        #c: exploration parameter
        #forced_rollout: whether to return inf score for unexplored node or not
        self.c = c
        self.forced_rollout = forced_rollout
        
    #implement
    def evaluate(self, node: Node):
        if node.n == 0 and self.forced_rollout:
            return float("inf")
        u = self.c * node.last_prob * sqrt(node.parent.n) / (1 + node.n)
        return node.mean_r + u