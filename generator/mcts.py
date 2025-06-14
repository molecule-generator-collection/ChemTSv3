import logging
from typing import Type, Self, Any
import numpy as np
from filter import Filter
from generator import Generator
from node import Node
from policy import Policy, UCB
from reward import Reward, LogPReward
from transition import WeightedTransition
from utils import class_from_class_path

class MCTS(Generator):
    def __init__(self, root: Node, transition: WeightedTransition, max_length=None, output_dir="generation_result", name=None, reward: Reward=LogPReward(), policy: Policy=UCB(), filters: list[Filter]=None, filtered_reward: float=0, n_tries=1, n_rollouts=1, expansion_threshold=0.995, use_dummy_reward: bool=False, logger: logging.Logger=None, info_interval: int=1):
        """
        Tries to maximize the reward by MCTS search.

        Args:
            root: root node
            expansion_threshold: [0-1]. Ignore children with low transition probabilities in expansion based on this value
            n_tries: how many tries before using filtered_reward
            use_dummy_reward: If True, backpropagate value is fixed to 0, still calculates rewards and objective values
        """
        self.root = root
        self.max_length = max_length or transition.max_length()
        self.policy = policy
        self.expansion_threshold = expansion_threshold
        self.n_tries = n_tries
        self.n_rollouts = n_rollouts
        self.use_dummy_reward = use_dummy_reward
        super().__init__(transition=transition, output_dir=output_dir, name=name, reward=reward, filters=filters, filtered_reward=filtered_reward, logger=logger, info_interval=info_interval)
        self._expand(self.root)

    def _expand(self, node: Node):
        if node.is_terminal():
            return

        actions, nodes, probs = zip(*self.transition.transitions_with_probs(node, threshold=self.expansion_threshold))
        for a, n in zip(actions, nodes):
            node.add_child(a, n)

    def _eval(self, node: Node, n_tries=1):
        if node.is_terminal(): #TODO: check on rollout, not here
            objective_values, reward = self.grab_objective_values_and_reward(node)
            node.sum_r = node.mean_r = -float("inf")
            return
        if not node.children: # if empty
            self._expand(node)
        
        to_backpropagate = []
        for _ in range(self.n_rollouts):
            for _ in range(self.n_tries):
                child = node.sample_node()
                objective_values, reward = self._rollout(child)
                if objective_values[0] != -float("inf"): # not filtered
                    break
            if objective_values[0] != -float("inf"):
                to_backpropagate.append((child, reward))
        if not to_backpropagate:
            self._backpropagate(node, self.filtered_reward, self.use_dummy_reward)
        for child, reward in to_backpropagate:
            self._backpropagate(child, reward, self.use_dummy_reward)

    def _rollout(self, node: Node):
        if node.depth >= self.max_length:
            return self.filtered_reward
        result = self.transition.rollout(node)
        return self.grab_objective_values_and_reward(result)

    def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
        while node:
            node.observe(0 if use_dummy_reward else value)
            node = node.parent

    # implement
    def _generate_impl(self):
        node = self.root
        while node.children:
            node = max(node.children.values(), key=lambda n: self.policy.evaluate(n))
            if node.sum_r == -float("inf"): # already exhausted every terminal under this node
                self.logger.debug("Exhausted every terminal under: " + str(node.parent) + "")
                node.parent.sum_r = node.parent.mean_r = -float("inf")
                node = self.root
        self._eval(node)