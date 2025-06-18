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
    def __init__(self, root: Node, transition: Transition, max_length=None, output_dir="generation_result", name=None, reward: Reward=LogPReward(), policy: Policy=UCB(), filters: list[Filter]=None, filtered_reward: float=-1, n_tries=1, n_rollouts=1, expansion_threshold=0.995, terminal_reward: float | str=None, freeze_terminal: bool=False, use_dummy_reward: bool=False, logger: logging.Logger=None, info_interval: int=1):
        """
        Tries to maximize the reward by MCTS search.

        Args:
            root: root node
            expansion_threshold: ([0,1]) ignore children with low transition probabilities in expansion based on this value
            n_rollouts: the number of rollouts in one step
            n_tries: the number of attempts to obtain an unfiltered node in a single rollout
            terminal_reward: If None, uses filtered_reward instead. If "reward", backpropagate the reward. If "ignore", doesn't backpropagate anything.
            freeze_terminal: If True, terminal node won't be visited twice.
            use_dummy_reward: If True, backpropagate value is fixed to 0. (still calculates rewards and objective values)
        """
        self.root = root
        self.max_length = max_length or transition.max_length()
        self.policy = policy
        self.n_tries = n_tries
        self.n_rollouts = n_rollouts
        self.expansion_threshold = expansion_threshold
        self.use_dummy_reward = use_dummy_reward
        if terminal_reward is None:
            terminal_reward = filtered_reward
        self.terminal_reward = terminal_reward
        self.freeze_terminal = freeze_terminal
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
        if node.n == 0:
            node = node.parent
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
            if self.terminal_reward != "ignore":
                if self.terminal_reward != "reward":
                    reward = self.terminal_reward
                self._backpropagate(node, reward, False)
            if self.freeze_terminal:
                node.sum_r = -float("inf")
            return
        
        if not node.children:
            self._expand(node)
            
        children = []
        for _ in range(self.n_rollouts):
            children.append(node.sample_child())

        for child in children:
            for _ in range(self.n_tries):
                objective_values, reward = self._rollout(child) # rollout returns the child itself if terminal
                if objective_values[0] != -float("inf"): # not filtered
                    break
            if objective_values[0] != -float("inf"):
                self._backpropagate(child, reward, self.use_dummy_reward)
            else:            
                self._backpropagate(child, self.filtered_reward, self.use_dummy_reward)