from math import sqrt
from node import Node
from policy import Policy

class PUCT(Policy):
    #implement
    def evaluate(node: Node, c=1, forced_rollout=True):
        #c: exploration parameter
        #forced_rollout: whether to return inf score for unexplored node or not
        if node.n == 0 and forced_rollout:
            return float("inf")
        u = c * node.last_prob * sqrt(node.parent.n) / (1 + node.n)
        return node.mean_r + u