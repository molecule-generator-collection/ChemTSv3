import logging
from typing import Type, Self, Any
import numpy as np
from filter import Filter
from generator import Generator
from node import Node
from policy import Policy, UCB1
from reward import Reward, LogPReward
from transition import Transition
from utils import class_from_class_path

class MCTS(Generator):
    def __init__(self, root: Node, transition: Transition, max_tree_depth=None, output_dir="generation_result", name=None, reward: Reward=LogPReward(), policy: Policy=UCB1(), filters: list[Filter]=None, filtered_reward: float | str | list=0, all_filtered_reward: float | str="ignore", rollout_width: int=1, allow_rollout_overlaps: bool=False, n_rollouts: int=1, n_tries: int =1, remove_failed_child: bool=False, terminal_reward: float | str="ignore", freeze_terminal: bool=True, use_dummy_reward: bool=False, logger: logging.Logger=None, info_interval: int=100):
        """
        Tries to maximize the reward by MCTS search.

        Args:
            root: root node
            rollout_width: the number of children to sample for rollout (to rollout all children, set this to higher values than the number of the tokens)
            allow_rollout_overlaps: whether to allow overlap nodes when sampling rollout candidates
            n_rollouts: the number of rollouts in one step
            n_tries: the number of attempts to obtain an unfiltered node in a single rollout
            remove_failed_child: If True, child nodes are will be removed when {n_rollouts * n_tries} rollouts are filtered.
            terminal_reward: If "ignore", doesn't backpropagate anything. If "reward", backpropagate the reward. If float value, backpropagate specified value.
            filtered_reward: Backpropagate this value when {n_tries} rollouts are filtered from the child. Set "ignore" not to backpropagate.
            all_filtered_reward: Backpropagate this value when {rollout_width * n_rollouts * n_tries} rollouts are filtered from the node.
            freeze_terminal: If True, terminal node won't be visited twice.
            use_dummy_reward: If True, backpropagate value is fixed to 0. (still calculates rewards and objective values)
        """
        self.root = root
        self.max_tree_depth = max_tree_depth or transition.max_length()
        self.policy = policy
        self.rollout_width = rollout_width
        self.allow_rollout_overlaps = allow_rollout_overlaps
        self.n_rollouts = n_rollouts
        self.n_tries = n_tries
        self.remove_failed_child = remove_failed_child
        if not isinstance(terminal_reward, (float, int)) and terminal_reward not in ("ignore", "reward"):
            raise ValueError("terminal_reward must be one of the following: float value, 'ignore', or 'reward'.")
        if terminal_reward == "ignore" and not freeze_terminal:
            raise ValueError("Set freeze_terminal to True, or set terminal_reward to something else.")
        if remove_failed_child and allow_rollout_overlaps:
            raise ValueError("Set one of these values to False: remove_failed_child or allow_rollout_overlaps.")
        if type(filtered_reward) is list and n_tries != 1:
            raise ValueError("list input for filtered_reward is not supported on n_tries > 1.")
        self.terminal_reward = terminal_reward
        self.freeze_terminal = freeze_terminal
        self.use_dummy_reward = use_dummy_reward
        self.all_filtered_reward = all_filtered_reward
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
        result = self.transition.rollout(node)
        return self.get_objective_values_and_reward(result)

    def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
        while node:
            node.observe(0 if use_dummy_reward else value)
            node = node.parent

    # implement
    def _generate_impl(self):
        node = self._selection()
        if node.is_terminal() or node.depth > self.max_tree_depth:
            if self.terminal_reward != "ignore":
                reward = self.terminal_reward
                self._backpropagate(node, self.terminal_reward, False)
            if self.freeze_terminal:
                node.n += 1 # to avoid n=0 score
                node.sum_r = -float("inf")
            return
        
        if not node.children and node.n != 0:
            self._expand(node)

        if not node.children:
            children = [node]
        else:
            children = node.sample_children(max_size=self.rollout_width, replace=self.allow_rollout_overlaps)
        
        parent_got_unfiltered_node = False
        for child in children:
            child_got_unfiltered_node = False
            for _ in range(self.n_rollouts):
                for _ in range(self.n_tries):
                    objective_values, reward = self._rollout(child) # rollout returns the child itself if terminal
                    if type(objective_values[0]) != str: # not filtered
                        break
                if type(objective_values[0]) != str: # not filtered
                    child_got_unfiltered_node = parent_got_unfiltered_node = True
                    self._backpropagate(child, reward, self.use_dummy_reward)
                elif self.filtered_reward[int(objective_values[0])] != "ignore":
                    self._backpropagate(child, self.filtered_reward[int(objective_values[0])], False)
            if self.remove_failed_child and not child_got_unfiltered_node:
                del child.parent.children[child.last_action]
        if self.all_filtered_reward != "ignore" and not parent_got_unfiltered_node:
            self._backpropagate(node, self.all_filtered_reward, False)
            self.logger.debug("All rollouts failed from: " + str(node))