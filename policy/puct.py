from math import sqrt
from node import Node
from policy import UCT

class PUCT(UCT):
    # override
    def get_exploration_term(self, node: Node):
        c = self.get_c_value(node)
        _, prior_weight = self.get_prior(node)
        n = node.n + prior_weight
        parent_n = node.parent.n + prior_weight
        return c * node.last_prob * sqrt(parent_n) / (1 + n)