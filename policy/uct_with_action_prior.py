from typing import Any, Callable
from node import Node
from policy import UCT
    
class UCTAP(UCT):
    """UCT with Action Prior"""
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, best_rate: float=0.0, prior_offset: float=0.2, prior_weight: int=1, max_prior: float=None):
        self.action_n = {}
        self.action_sum_r = {}
        self.prior_offset = prior_offset
        super().__init__(c=c, best_rate=best_rate, prior_weight=prior_weight, max_prior=max_prior)

    # override
    def observe(self, parent: Node, action: Any, child: Node, objective_values: list[float], reward: float):
        if type(action) == tuple:
            action = action[0]
        
        if not action in self.action_n:
            self.action_n[action] = 1
            self.action_sum_r[action] = reward
        else:
            self.action_n[action] += 1
            self.action_sum_r[action] += reward

    # override
    def get_prior(self, node: Node) -> tuple[float, int]:
        if not node.last_action in self.action_n:
            return 0, 0
        else:
            action_mean_r = self.action_sum_r[node.last_action] / self.action_n[node.last_action]
            return action_mean_r + self.prior_offset, self.prior_weight
        