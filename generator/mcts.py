import pickle
import time
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
    def __init__(self, root: Node, transition: WeightedTransition, max_length=None, output_dir="generation_result", name=None, reward: Reward=LogPReward(), policy: Policy=UCB(), filters: list[Filter]=None, filtered_reward: float=0, logger_conf: dict[str, Any]=None, n_tries=1, expansion_threshold=0.995, exhaust_backpropagate: bool=False, use_dummy_reward: bool=False, rollout_conf = None):
        """
        Tries to maximize the reward by MCTS search.

        Args:
            root: root node
            expansion_threshold: [0-1]. Ignore children with low transition probabilities in expansion based on this value
            n_tries: how many tries before using filtered_reward
            exhaust_backpropagate: If true, backpropagate the reward when every terminal node under the node is already explored (only once, as that node won't be visited again)
            use_dummy_reward: If True, backpropagate value is fixed to 0, still calculates rewards and objective values
            rollout_conf: config for rollout.
        """
        self.root = root
        self.transition = transition
        self.max_length = max_length or transition.max_length()
        self.policy = policy
        self.expansion_threshold = expansion_threshold
        self.n_tries = n_tries
        self.exhaust_backpropagate = exhaust_backpropagate
        self.use_dummy_reward = use_dummy_reward
        self.rollout_conf = rollout_conf or {"rollout_threshold": self.expansion_threshold}
        self.rollout_count = 0 #unused
        self._expand(self.root)
        super().__init__(output_dir=output_dir, name=name, reward=reward, filters=filters, filtered_reward=filtered_reward, logger_conf=logger_conf)

    def _expand(self, node: Node):
        if node.is_terminal():
            return

        actions, nodes, probs = zip(*self.transition.transitions_with_probs(node, threshold=self.expansion_threshold))
        for a, n in zip(actions, nodes):
            node.add_child(a, n)

    def _eval(self, node: Node):
        if node.is_terminal():
            objective_values, reward = self.grab_objective_values_and_reward(node)
            node.sum_r = node.mean_r = -float("inf")
            return objective_values, reward
        if not bool(node.children): # if empty
            self._expand(node)
        objective_values, reward = self._rollout(node)
        return objective_values, reward

    def _rollout(self, node: Node):
        # TODO: change here
        if node.depth >= self.max_length:
            return self.filtered_reward
        result = self.transition.rollout(node, **self.rollout_conf)
        self.rollout_count += 1
        return self.grab_objective_values_and_reward(result)

    def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
        while node:
            if use_dummy_reward:
                node.observe(0)
            else:
                node.observe(value)
            node = node.parent

    # implement
    def _generate_impl(self):
        node = self.root
        while node.children:
            node = max(node.children.values(), key=lambda n: self.policy.evaluate(n))
            if node.sum_r == -float("inf"): # already exhausted every terminal under this
                self.logger.debug("Exhausted every terminal under: " + str(node.parent) + "")
                if self.exhaust_backpropagate:
                    value = self._eval(node)[1]
                    self._backpropagate(node, value, self.use_dummy_reward)
                node.parent.sum_r = node.parent.mean_r = -float("inf")
                node = self.root
                continue
        for _ in range(self.n_tries):
            objective_values, reward = self._eval(node)
            if objective_values[0] != -float("inf"): # not filtered
                break
        self._backpropagate(node, reward, self.use_dummy_reward)

    # override
    def name(self):
        if self._name is not None:
            return self._name
        else:
            policy_name = self.policy.name()
            return super().name() + "_" + policy_name

    def save(self, file: str):
        with open(file, mode="wb") as fo:
            pickle.dump(self.root, fo)
            pickle.dump(self._name, fo)
            pickle.dump(self._output_dir, fo)
            pickle.dump(self.unique_keys, fo)
            pickle.dump(self.record, fo)
            pickle.dump(self.rollout_count, fo)
            pickle.dump(self.passed_time, fo)
            pickle.dump(self.reward, fo)
            pickle.dump(self.policy, fo)
            pickle.dump(self.filters, fo)
            pickle.dump(self.filtered_reward, fo)
    
    # transition won't be saved/loaded
    @staticmethod
    def load(file: str, transition: WeightedTransition) -> Self:
        with open(file, "rb") as f:
            root = pickle.load(f)
            s = MCTS(root=root, transition=transition)
            s._name = pickle.load(f)
            s._output_dir = pickle.load(f)
            s.unique_keys = pickle.load(f)
            s.record = pickle.load(f)
            s.rollout_count = pickle.load(f)
            s.passed_time = pickle.load(f)
            s.reward = pickle.load(f)
            s.policy = pickle.load(f)
            s.filters = pickle.load(f)
            s.filtered_reward = pickle.load(f)
        return s