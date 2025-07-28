from math import sqrt
from node import Node
from policy import UCT

class PUCT(UCT):
    # override
    def get_exploration_term(self, node: Node):
        c = self.get_c_value(node)
        return c * node.last_prob * sqrt(node.parent.n) / (1 + node.n)