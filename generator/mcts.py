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
from utils import get_class_from_class_path

class MCTS(Generator):
    def __init__(self, transition: WeightedTransition, max_length=None, output_dir="generation_result", name=None, reward: Reward=LogPReward(), policy: Policy=UCB(), filters: list[Filter]=None, logger_conf: dict[str, Any]=None):
        #name: if you plan to change the policy_class or policy_class's c value, you might want to set the name manually
        self.root = None
        self.transition = transition
        self.max_length = max_length or transition.max_length()
        self.policy = policy
        self.count_rollouts = 0
        self.passed_time = 0
        #for search
        self.expansion_threshold = 0.995
        self.rollout_conf = {"rollout_threshold": self.expansion_threshold}
        super().__init__(output_dir=output_dir, name=name, reward=reward, filters=filters, logger_conf=logger_conf)

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
            return reward
        if not bool(node.children): #if empty
            self._expand(node)
        objective_values, reward = self._rollout(node)
        return reward

    def _rollout(self, node: Node):
        #TODO: change here
        if node.depth >= self.max_length:
            return self.reward.filtered_reward
        result = self.transition.rollout(node, **self.rollout_conf)
        self.count_rollouts += 1
        return self.grab_objective_values_and_reward(result)

    def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
        while node:
            if use_dummy_reward:
                node.observe(0)
            else:
                node.observe(value)
            node = node.parent

    #implement
    def generate(self, root: Node=None, time_limit: float=None, max_generations: int=None, max_rollouts: int=None, expansion_threshold: float=0.995, exhaust_backpropagate: bool=False, use_dummy_reward: bool=False, change_root: bool=False, rollout_conf: dict[str, Any]=None):
        """
        Generate nodes that either is_terminal() = True or depth = max_length. Tries to maximize the reward by MCTS search.

        Args:
            time_limit: Seconds. Generation stops after the time limit.
            max_generations: Generation stops after generating 'max_generations' number of nodes.
            max_rollouts: Generation stops after conducting 'max_rollouts' number of rollouts.
            exhaust_backpropagate: If true, backpropagate the reward when every terminal node under the node is already explored (only once, as that node won't be visited again)
            expansion_threshold: [0-1]. Ignore children with low transition probabilities in expansion based on this value
            use_dummy_reward: If True, backpropagate value is fixed to 0, still calculates rewards and objective values
            change_root: Failsafe. Set to False only if you want to change the root node in the loaded generator.
            rollout_conf: config for rollout.
        """
        
        if (max_rollouts is None) and (time_limit is None) and (max_generations is None):
            raise ValueError("Specify at least one of max_genrations, max_rollouts or time_limit.")

        #refresh variables
        self.expansion_threshold = expansion_threshold or self.expansion_threshold
        self.rollout_conf = rollout_conf or self.rollout_conf

        #record current time and counts
        time_start = time.time()
        initial_time = self.passed_time
        initial_count_rollouts = self.count_rollouts
        initial_count_generations = len(self.unique_keys)

        #asign root node
        if root is None and self.root is None:
            raise ValueError("Specify the root node unless you're running a loaded MCTS searcher.")
        if (root is not None) and (self.root is not None) and not change_root:
            raise ValueError("root was passed as an argument, but this MCTS searcher already has the root node. If you really want to change the root node, set change_root to True.")

        if (root is not None and change_root) or self.root is None:
            self.root = root
            self._expand(self.root)

        self.logger.info("Search is started.")
        while True:
            time_passed = time.time() - time_start
            self.passed_time = initial_time + time_passed
            if time_limit is not None and time_passed >= time_limit:
                break
            if max_rollouts is not None and self.count_rollouts - initial_count_rollouts >= max_rollouts:
                break
            if max_generations is not None and len(self.unique_keys) - initial_count_generations >= max_generations:
                break

            node = self.root
            while node.children:
                node = max(node.children.values(), key=lambda n: self.policy.evaluate(n))
                if node.sum_r == -float("inf"): #already exhausted every terminal under this
                    self.logger.debug("Exhausted every terminal under: " + str(node.parent) + "")
                    if exhaust_backpropagate:
                        value = self._eval(node)
                        self._backpropagate(node, value, use_dummy_reward)
                    node.parent.sum_r = node.parent.mean_r = -float("inf")
                    node = self.root
                    continue
            value = self._eval(node)
            self._backpropagate(node, value, use_dummy_reward)
            
        self.logger.info("Search is completed.")

    def log_unique_node(self, key, objective_values, reward):
        self.logger.info(str(len(self.unique_keys)) + "- time: " + "{:.2f}".format(self.passed_time) + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(reward) + ", node: " + key)
        self.unique_keys.append(key)
        self.record[key] = {}
        self.record[key]["objective_values"] = objective_values
        self.record[key]["reward"] = reward
        self.record[key]["count_rollouts"] = self.count_rollouts
        self.record[key]["time"] = self.passed_time
        self.record[key]["generation_order"] = len(self.unique_keys)

    def grab_objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
        key = str(node)
        if key in self.record:
            self.logger.debug("Already in dict: " + key + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(self.record[key]["reward"]))
            return self.record[key]["objective_values"], self.record[key]["reward"]
        
        for filter in self.filters:
            if not filter.check(node):
                self.logger.debug("filtered by " + filter.__class__.__name__ + ": " + key)
                return ([0,0], self.reward.filtered_reward)
            
        objective_values, reward = self.reward.objective_values_and_reward(node)
        self.log_unique_node(key, objective_values, reward)

        return objective_values, reward
    
    #override
    def name(self):
        if self._name is not None:
            return self._name
        else:
            policy_name = self.policy.name()
            return super().name() + "_" + policy_name

    def save(self, file: str):
        with open(file, mode="wb") as fo:
            pickle.dump(self._name, fo)
            pickle.dump(self._output_dir, fo)
            pickle.dump(self.root, fo)
            pickle.dump(self.unique_keys, fo)
            pickle.dump(self.record, fo)
            pickle.dump(self.count_rollouts, fo)
            pickle.dump(self.passed_time, fo)
            pickle.dump(self.reward, fo)
            pickle.dump(self.policy, fo)
    
    #transition won't be saved/loaded
    @staticmethod
    def load(file: str, transition: WeightedTransition) -> Self:
        s = MCTS(transition=transition)
        with open(file, "rb") as f:
            s._name = pickle.load(f)
            s._output_dir = pickle.load(f)
            s.root = pickle.load(f)
            s.unique_keys = pickle.load(f)
            s.record = pickle.load(f)
            s.count_rollouts = pickle.load(f)
            s.passed_time = pickle.load(f)
            s.reward = pickle.load(f)
            s.policy = pickle.load(f)
        return s