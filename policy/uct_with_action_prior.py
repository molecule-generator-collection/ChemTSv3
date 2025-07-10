from math import log, sqrt
from typing import Any, Callable
from node import Node
from policy import UCT
    
class UCTAP(UCT):
    """UCT with Action Prior"""
    def __init__(self, c: Callable[[float], float] | list[tuple[float, float]] | float=1, best_rate: float=0.0, c_action:float=0.1, prior_offset: float=0.1, prior_weight: int=1, no_prior_for_unvisited: bool=True, max_prior: float=None):
        self.sum_action_n = 0
        self.action_n = {}
        self.action_sum_r = {}
        self.c_action = c_action
        self.prior_offset = prior_offset
        super().__init__(c=c, best_rate=best_rate, prior_weight=prior_weight, no_prior_for_unvisited=no_prior_for_unvisited, max_prior=max_prior)

    # override
    def observe(self, child: Node, objective_values: list[float], reward: float):
        self.sum_action_n += 1
        action = child.last_action
        if type(action) == tuple:
            action = action[0]
            
        if child.parent.reward is not None:
            reward_dif = reward - child.parent.reward
        else:
            return
        
        if not action in self.action_n:
            self.action_n[action] = 1
            self.action_sum_r[action] = reward_dif
        else:
            self.action_n[action] += 1
            self.action_sum_r[action] += reward_dif

    # override
    def get_prior(self, node: Node) -> float:
        last_action = node.last_action
        if type(last_action) == tuple:
            last_action = last_action[0]
            
        if not last_action in self.action_n or node.parent.reward is None:
            return None
        else:
            prior = node.parent.reward
            prior += self.action_sum_r[last_action] / self.action_n[last_action]
            prior += self.c_action * sqrt(log(self.sum_action_n) / (self.action_n[last_action]))
            prior += self.prior_offset
            return prior