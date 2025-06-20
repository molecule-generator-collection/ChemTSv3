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
    def __init__(self, root: Node, transition: Transition, max_length=None, output_dir="generation_result", name=None, reward: Reward=LogPReward(), policy: Policy=UCB(), filters: list[Filter]=None, filtered_reward: float=0, rollout_width: int=1, allow_rollout_overlaps: bool=False, n_rollouts: int=1, n_tries: int =1, terminal_reward: float | str="ignore", freeze_terminal: bool=True, use_dummy_reward: bool=False, logger: logging.Logger=None, info_interval: int=100):
        """
        Tries to maximize the reward by MCTS search.

        Args:
            root: root node
            n_rollouts: the number of rollouts in one step
            n_tries: the number of attempts to obtain an unfiltered node in a single rollout
            cut_unvisited_children: 
            terminal_reward: If "ignore", doesn't backpropagate anything. If "reward", backpropagate the reward. If float value, backpropagate specified value.
            freeze_terminal: If True, terminal node won't be visited twice.
            use_dummy_reward: If True, backpropagate value is fixed to 0. (still calculates rewards and objective values)
        """
        self.root = root
        self.max_length = max_length or transition.max_length()
        self.policy = policy
        self.rollout_width = rollout_width
        self.allow_rollout_overlaps = allow_rollout_overlaps
        self.n_rollouts = n_rollouts
        self.n_tries = n_tries
        if not isinstance(terminal_reward, (float, int)) and terminal_reward not in ("ignore", "reward"):
            raise ValueError("terminal_reward must be one of the following: float value, 'ignore', or 'reward'.")
        if terminal_reward == "ignore" and not freeze_terminal:
            raise ValueError("Set freeze_terminal to True, or set terminal_reward to something else.")
        self.terminal_reward = terminal_reward
        self.freeze_terminal = freeze_terminal
        self.use_dummy_reward = use_dummy_reward
        super().__init__(transition=transition, output_dir=output_dir, name=name, reward=reward, filters=filters, filtered_reward=filtered_reward, logger=logger, info_interval=info_interval)
        self.root.n = 1
        
    def _selection(self) -> Node:
        node = self.root
        while node.children:
            node = self.policy.select_child(node)
            if node.sum_r == -float("inf"): # already exhausted every terminal under this node
                self.logger.debug("Exhausted every terminal under: " + str(node.parent) + "")
                node.parent.sum_r = -float("inf")
                node = self.root
        return node

    def _expand(self, node: Node):
        actions, nodes, _ = zip(*self.transition.transitions_with_probs(node))
        for a, n in zip(actions, nodes):
            node.add_child(a, n)
            
    def _rollout(self, node: Node):
        if node.depth >= self.max_length:
            return ["0"], self.filtered_reward
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
                if type(self.terminal_reward) == float:
                    reward = self.terminal_reward
                self._backpropagate(node, reward, False)
            if self.freeze_terminal:
                node.sum_r = -float("inf")
            return
        
        if not node.children and node.n != 0:
            self._expand(node)

        if not node.children:
            children = [node]
        else:
            children = node.sample_children(max_size=self.rollout_width, replace=self.allow_rollout_overlaps)
        
        for child in children:
            got_unfiltered_node = False
            for _ in range(self.n_rollouts):
                for _ in range(self.n_tries):
                    objective_values, reward = self._rollout(child) # rollout returns the child itself if terminal
                    if type(objective_values[0]) != str: # not filtered
                        break
                if type(objective_values[0]) != str: # not filtered
                    got_unfiltered_node = True
                    self._backpropagate(child, reward, self.use_dummy_reward)
            if not got_unfiltered_node:
                self._backpropagate(child, self.filtered_reward, False)