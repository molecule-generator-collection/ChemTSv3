import pickle
import time
from typing import Type, Self, Any
import numpy as np
from generator import Generator
from node import Node
from policy import Policy, UCB
from reward import LogPReward
from transition import WeightedTransition
from utils import get_class_from_str

class MCTS(Generator):
  def __init__(self, transition: WeightedTransition, max_length=None, output_dir="result", name=None, reward_class_path: str="reward.logp_reward.LogPReward", objective_values_conf: dict[str, Any]=None, reward_conf: dict[str, Any]=None, policy_class_path: str="policy.ucb.UCB", policy_conf: dict[str, Any]=None, logger_conf: dict[str, Any]=None):
    #name: if you plan to change the policy_class or policy_class's c value, you might want to set the name manually
    self.root = None
    self.transition = transition
    self.policy_class: Type[Policy] = get_class_from_str(policy_class_path)
    self.policy_conf = policy_conf or {}
    self.max_length = max_length or transition.max_length()
    self.count_rollouts = 0
    self.passed_time = 0
    #for search
    self.expansion_threshold = 0.995
    self.rollout_conf = {"rollout_threshold": self.expansion_threshold}
    super().__init__(output_dir=output_dir, name=name, reward_class_path=reward_class_path, objective_values_conf=objective_values_conf, reward_conf=reward_conf, logger_conf=logger_conf)

  #override
  def name(self):
    if self._name is not None:
      return self._name
    else:
      policy_name = self.policy_class.__name__
      policy_c = str(self.policy_conf.get("c", 1))
      return super().name() + "_" + policy_name + "_c=" + policy_c + "_"

  def _expand(self, node: Node):
    if node.is_terminal():
      return

    #apply expansion_threshold
    actions, nodes, probs = zip(*self.transition.transitions_with_probs(node))
    remaining_indices = MCTS.select_indices_by_threshold(probs, self.expansion_threshold)

    for idx in remaining_indices:
      node.add_child(actions[idx], nodes[idx])

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
    if node.depth >= self.max_length:
      return self.reward_conf.get("filtered_reward", -1)
    mol = self.transition.rollout(node, **self.rollout_conf)
    self.count_rollouts += 1
    return self.grab_objective_values_and_reward(mol)

  def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
    while node:
      if use_dummy_reward:
        node.observe(0)
      else:
        node.observe(value)
      node = node.parent

  #implement
  def generate(self, root: Node=None, time_limit=None, max_generations=None, max_rollouts=None, use_dummy_reward=False, expansion_threshold=0.995, exhaust_backpropagate=False, change_root=False, rollout_conf: dict[str, Any]=None):
    #exhaust_backpropagate: whether to backpropagate or not when every terminal node under the node is already explored (only once: won't be visited again)
    #expansion_threshold: [0-1], ignore children with low transition probabilities in expansion based on this value
    #rollout_conf: config for rollout
    #dummy_reward: backpropagate value is fixed to 0, still calculates rewards and objective values
    
    if (max_rollouts is None) and (time_limit is None) and (max_generations is None):
        raise AssertionError("Specify at least one of max_genrations, max_rollouts or time_limit.")

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
      raise AssertionError("Specify the root node unless you're running a loaded MCTS searcher.")
    if (root is not None) and (self.root is not None) and not change_root:
      raise AssertionError("root was passed as an argument, but this MCTS searcher already has the root node. If you really want to change the root node, set change_root to True.")

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
        node = max(node.children.values(), key=lambda n: self.policy_class.evaluate(n, **self.policy_conf))
        if node.sum_r == -float("inf"): #already exhausted every terminal under this
          self.logger.debug("Exhaust every terminal under: " + str(node.parent) + "")
          if exhaust_backpropagate:
            value = self._eval(node)
            self._backpropagate(node, value, use_dummy_reward)
          node.parent.sum_r = node.parent.mean_r = -float("inf")
          node = self.root
          continue
      value = self._eval(node)
      self._backpropagate(node, value, use_dummy_reward)

    self.plot_objective_values_and_reward(x_axis = "generation_order", maxline = True)
    self.plot_objective_values_and_reward(x_axis = "time", maxline = True)
    self.logger.info("Search is completed.")

  #for expansion_threshold
  #move to WeightedTransition later
  @staticmethod
  def select_indices_by_threshold(probs: list[float], expansion_threshold: float) -> list[int]:
    probs = np.array(probs)
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, expansion_threshold)
    return sorted_indices[:cutoff + 1].tolist()

  def log_unique_mol(self, key, objective_values, reward):
    self.logger.info(str(len(self.unique_keys)) + "- time: " +  "{:.2f}".format(self.passed_time) + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(reward) + ", mol: " + key)
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
      self.logger.debug("already in dict: " + key + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(self.record[key]["reward"]))
      return self.record[key]["objective_values"], self.record[key]["reward"]
    objective_values, reward = self.reward_class.objective_values_and_reward(node, objective_values_conf=self.objective_values_conf, reward_conf=self.reward_conf)

    if hasattr(node, "is_valid_mol") and callable(getattr(node, "is_valid_mol")) and not node.is_valid_mol(): #if node has is_valid_mol() method, check whether valid or not
      self.logger.debug("invalid mol: " + key)
    else:
      self.log_unique_mol(key, objective_values, reward)

    return objective_values, reward
  
  def save(self, file: str):
    self._name
    with open(file, mode="wb") as fo:
      pickle.dump(self._name, fo)
      pickle.dump(self._output_dir, fo)
      pickle.dump(self.root, fo)
      pickle.dump(self.unique_keys, fo)
      pickle.dump(self.record, fo)
      pickle.dump(self.count_rollouts, fo)
      pickle.dump(self.passed_time, fo)
      pickle.dump(self.reward_class.__name__, fo)
      pickle.dump(self.objective_values_conf, fo)
      pickle.dump(self.reward_conf, fo)
      pickle.dump(self.policy_class.__name__, fo)
      pickle.dump(self.policy_conf, fo)
  
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
      
      reward_name = pickle.load(f)
      reward_class = globals().get(reward_name, None)
      if reward_class is None:
        s.logger.warning("Reward class " + reward_name + " was not found, and replaced with LogPReward.")
        s.reward_class = LogPReward
      else:
        s.reward_class = reward_class
      s.objective_values_conf = pickle.load(f)
      s.reward_conf = pickle.load(f)
      
      policy_name = pickle.load(f)
      policy_class = globals().get(policy_name, None)
      if policy_class is None:
        s.logger.warning("Policy class " + policy_name + " was not found, and replaced with UCB.")
        s.policy_class = UCB
      else:
        s.policy_class = policy_class
      s.policy_conf = pickle.load(f)
    return s