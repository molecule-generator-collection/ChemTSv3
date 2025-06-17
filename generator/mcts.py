import logging
from typing import Type, Self, Any
import numpy as np
from filter import Filter
from generator import Generator
from node import Node
from policy import Policy, UCB
from reward import Reward, LogPReward
from transition import Transition
from utils import class_from_class_path

class MCTS(Generator):
    def __init__(self, root: Node, transition: Transition, max_length=None, output_dir="generation_result", name=None, reward: Reward=LogPReward(), policy: Policy=UCB(), filters: list[Filter]=None, filtered_reward: float=0, forced_rollout=True, n_tries=1, n_rollouts=1, expansion_threshold=0.995, use_dummy_reward: bool=False, logger: logging.Logger=None, info_interval: int=1):
        """
        Tries to maximize the reward by MCTS search.

        Args:
            root: root node
            expansion_threshold: ([0,1]) ignore children with low transition probabilities in expansion based on this value
            n_rollouts: the number of rollouts in one step
            n_tries: the number of attempts to obtain an unfiltered node in a single rollout
            use_dummy_reward: If True, backpropagate value is fixed to 0. (still calculates rewards and objective values)
        """
        self.root = root
        self.max_length = max_length or transition.max_length()
        self.policy = policy
        self.forced_rollout = forced_rollout
        self.n_tries = n_tries
        self.n_rollouts = n_rollouts
        self.expansion_threshold = expansion_threshold
        self.use_dummy_reward = use_dummy_reward
        super().__init__(transition=transition, output_dir=output_dir, name=name, reward=reward, filters=filters, filtered_reward=filtered_reward, logger=logger, info_interval=info_interval)
        self.root.n = 1
        
    def _selection(self) -> Node:
        node = self.root
        while node.children:
            node = max(node.children.values(), key=lambda n: self.policy.evaluate(n))
            if node.sum_r == -float("inf"): # already exhausted every terminal under this node
                self.logger.debug("Exhausted every terminal under: " + str(node.parent) + "")
                node.parent.sum_r = -float("inf")
                node = self.root
        return node

    def _expand(self, node: Node):
        actions, nodes, probs = zip(*self.transition.transitions_with_probs(node, threshold=self.expansion_threshold))
        for a, n in zip(actions, nodes):
            node.add_child(a, n)
            
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
        node = self._selection()
        if node.is_terminal():
            objective_values, reward = self.grab_objective_values_and_reward(node)
            node.sum_r = -float("inf")
            return
        if not node.children and node.n != 0:
            self._expand(node)
            
        if self.forced_rollout:
            children = list(node.children.values())
        elif node.n == 0:
            children = [node]
        else:
            children = [node.sample_child()]

        for child in children:
            to_backpropagate = []
            for _ in range(self.n_rollouts):
                for _ in range(self.n_tries):
                    objective_values, reward = self._rollout(child) # rollout returns the child itself if terminal
                    if objective_values[0] != -float("inf"): # not filtered
                        break
                if objective_values[0] != -float("inf"):
                    to_backpropagate.append((child, reward))
            if not to_backpropagate:
                self._backpropagate(child, self.filtered_reward, self.use_dummy_reward)
            for child, reward in to_backpropagate:
                self._backpropagate(child, reward, self.use_dummy_reward)