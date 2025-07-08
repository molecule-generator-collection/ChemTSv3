from typing import Any, Callable
from node import Node
from policy import UCT
    
class UCTAP(UCT):
    """UCT with Action Prior"""
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, best_rate: float=0.0, prior_offset: float=0.2, prior_weight: int=1, max_prior: float=None, use_parent_reward: bool=True):
        self.action_n = {}
        self.action_sum_r = {}
        self.prior_offset = prior_offset
        self.use_parent_reward = use_parent_reward
        super().__init__(c=c, best_rate=best_rate, prior_weight=prior_weight, max_prior=max_prior)

    # override
    def observe(self, child: Node, objective_values: list[float], reward: float):
        action = child.last_action
        if type(action) == tuple:
            action = action[0]
            
        if self.use_parent_reward and child.parent.reward is not None:
            reward = reward - child.parent.reward
        
        if not action in self.action_n:
            self.action_n[action] = 1
            self.action_sum_r[action] = reward
        else:
            self.action_n[action] += 1
            self.action_sum_r[action] += reward

    # override
    def get_prior(self, node: Node) -> tuple[float, int]:
        last_action = node.last_action
        if type(last_action) == tuple:
            last_action = last_action[0]
            
        if not last_action in self.action_n:
            return 0, 0
        else:
            prior = self.action_sum_r[last_action] / self.action_n[last_action]
            prior += self.prior_offset
            if self.use_parent_reward and node.parent.reward is not None:
                prior += node.parent.reward
            return prior, self.prior_weight
        