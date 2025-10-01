from math import sqrt
from node import Node
from policy import UCT

class PUCT(UCT):
    """
    Modified PUCT based on AlphaZero.
    Args:
        c: The weight of the exploration term. Higher values place more emphasis on exploration over exploitation.
        best_rate: A value between 0 and 1. The exploitation term is computed as 
                    best_rate * (best reward) + (1 - best_rate) * (average reward).
        max_prior: A lower bound for the best reward. If the actual best reward is lower than this value, this value is used instead.
        pw_c: Used for progressive widening.
        pw_alpha: Used for progressive widening.
    """
    # override
    def get_exploration_term(self, node: Node):
        c = self.get_c_value(node)
        return c * node.last_prob * sqrt(node.parent.n) / (1 + node.n)