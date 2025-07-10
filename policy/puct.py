from math import sqrt
from node import Node
from policy import UCT

class PUCT(UCT):
    # override
    def get_exploration_term(self, node: Node):
        c = self.get_c_value(node)
        _, prior_weight = self.get_prior(node)
        n = max(node.n + prior_weight, 1)
        parent_n = max(node.parent.n + prior_weight, 1)
        return c * node.last_prob * sqrt(parent_n) / (1 + n)