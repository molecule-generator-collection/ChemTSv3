from math import log, sqrt
from node import Node
from policy import Policy

class UCB(Policy):
    def __init__(self, c: float=1):
        self.c = c
    
    #implement
    def evaluate(self, node: Node):
        #c: exploration parameter
        if node.n == 0:
            return float("inf")
        u = self.c * sqrt(2 * log(node.parent.n) / (node.n))
        return node.mean_r + u