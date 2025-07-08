from math import sqrt
from policy import UCT

class PUCT(UCT):
    # override
    def get_exploration_term(self, c: float, parent_n: int, n: int, last_prob: float):
        return c * last_prob * sqrt(parent_n) / (1 + n)