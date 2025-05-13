import time, datetime
from typing import Type
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from node.node import Node
from edgepredictor.edgepredictor import EdgePredictor
from policy.policy import Policy
from policy.ucb import UCB
from reward.reward import Reward
from reward.logp_reward import LogP_reward
from .searcher import Searcher

class MCTS(Searcher):
  def __init__(self, edgepredictor: EdgePredictor, rewardfunc: Type[Reward] = LogP_reward, reward_conf: dict = None, policy: Type[Policy] = UCB, policy_conf: dict = None, rollout_limit=4096, print_output=True, verbose=False, name=None):
    self.edgepredictor = edgepredictor
    self.rewardfunc = rewardfunc
    self.reward_conf = reward_conf or {}
      #null_reward ...  reward for invalid molecules, default: 1
    self.policy = policy
    self.policy_conf = policy_conf or {}
    self.rollout_limit = rollout_limit
    self.verbose = verbose
    self.count_rollouts = 0
    self.passed_time = 0
    #for search
    self.expansion_threshold = 0.995
    self.rollout_threshold = 0.995
    super().__init__(name, print_output=print_output)
  
  #override
  def name(self):
    if self._name is not None:
      return self._name
    else:
      policy_name = self.policy.__name__
      policy_c = str(self.policy_conf.get("c", 1))
      newname = policy_name + ", c = " + policy_c + ", " + str(datetime.datetime.now())
      return newname

  def _expand(self, node: Node):
    if node.is_terminal():
      return

    #apply expansion_threshold
    nodes = self.edgepredictor.nextnodes_with_probs(node)
    probs = [node.lastprob for node in nodes]
    remaining_ids = MCTS.select_indices_by_threshold(probs, self.expansion_threshold)

    for id in remaining_ids:
      node.children[id] = nodes[id]

  def _eval(self, node: Node):
    if node.is_terminal():
      objective_values, reward = self.grab_objective_values_and_reward(node)
      node.sum_r = node.mean_r = -float("inf")
      return reward
    if not bool(node.children): #if empty
      self._expand(node)
    objective_values, reward = self._rollout(node)
    return reward

  def _rollout(self, node):
    if node.idtensor.numel() >= self.rollout_limit:
      return self.reward_conf.get("null_reward", -1)
    mol = self.edgepredictor.randomgen(node, conf={"rollout_threshold": self.rollout_threshold})
    self.count_rollouts += 1
    return self.grab_objective_values_and_reward(mol)

  def _backpropagate(self, node, value):
    while node:
      node.n += 1
      node.sum_r += value
      node.mean_r = node.sum_r / node.n
      node = node.parent

  def search(self, root: Node, expansion_threshold=None, rollout_threshold=None, exhaust_backpropagate=False, dummy_reward=False, max_rollouts=None, time_limit=None, max_generations=None):
    #exhaust_backpropagate: whether to backpropagate or not when every terminal node under the node is already explored (only once: won't be visited again)
    #expansion_threshold: [0-1], ignore children with low transition probabilities in expansion based on this value
    #rollout_threshold: [0-1], ignore children with low transition probabilities in rollout based on this value, set to the same value as expansion_threshold by default
    #dummy_reward: backpropagate value is fixed to 0, still calculates rewards and objective values
    
    assert (max_rollouts is not None) or (time_limit is not None) or (max_generations is not None), \
        "specify at least one of max_genrations, max_rollouts or time_limit"

    if expansion_threshold is not None:
      self.expansion_threshold = expansion_threshold
      if rollout_threshold is None:
        self.rollout_threshold = expansion_threshold
    if rollout_threshold is not None:
      self.rollout_threshold = rollout_threshold

    time_start = time.time()
    initial_time = self.passed_time
    initial_count_rollouts = self.count_rollouts
    initial_count_generations = len(self.unique_molkeys)

    self._expand(root)

    while True:
      time_passed = time.time() - time_start
      self.passed_time = initial_time + time_passed
      if time_limit is not None and time_passed >= time_limit:
        break
      if max_rollouts is not None and self.count_rollouts - initial_count_rollouts >= max_rollouts:
        break
      if max_generations is not None and len(self.unique_molkeys) - initial_count_generations >= max_generations:
        break

      node = root
      while node.children:
        node = max(node.children.values(), key=lambda n: self.policy.evaluate(n, conf=self.policy_conf))
        if node.sum_r == -float("inf"): #already exhausted every terminal under this
          if self.verbose:
            self.logging("!------exhaust every terminal under: " + str(node.parent) + "------!")
          if exhaust_backpropagate:
            value = self._eval(node)
            if dummy_reward:
              value = 0
            self._backpropagate(node, value)
          node.parent.sum_r = node.parent.mean_r = -float("inf")
          node = root
          continue
      value = self._eval(node)
      if dummy_reward:
        value = 0
      self._backpropagate(node, value)

    print("Search is completed.")

  #for expansion_threshold
  @staticmethod
  def select_indices_by_threshold(probs: list[float], expansion_threshold: float) -> list[int]:
      probs = np.array(probs)
      sorted_indices = np.argsort(-probs)
      sorted_probs = probs[sorted_indices]
      cumulative_probs = np.cumsum(sorted_probs)
      cutoff = np.searchsorted(cumulative_probs, expansion_threshold)
      return sorted_indices[:cutoff + 1].tolist()

  def log_unique_mol(self, key, objective_values, reward):
      self.logging(str(len(self.unique_molkeys)) + "- time: " +  "{:.2f}".format(self.passed_time) + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(reward) + ", mol: " + key)
      self.unique_molkeys.append(key)
      self.record[key] = {}
      self.record[key]["objective_values"] = objective_values
      self.record[key]["reward"] = reward
      self.record[key]["count_rollouts"] = self.count_rollouts
      self.record[key]["time"] = self.passed_time
      self.record[key]["generation_order"] = len(self.unique_molkeys)

  def grab_objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
    key = str(node)
    if key in self.record:
      if self.verbose:
        self.logging("already in dict: " + key + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(self.record[key]["reward"]))
      return self.record[key]["objective_values"], self.record[key]["reward"]
    objective_values, reward = self.rewardfunc.objective_values_and_reward(node, conf=self.reward_conf)

    if hasattr(node, "is_valid_mol") and callable(getattr(node, "is_valid_mol")) and not node.is_valid_mol(): #if node has is_valid_mol() method, check whether valid or not
      if self.verbose:
        self.logging("invalid mol: " + key)
    else:
      self.log_unique_mol(key, objective_values, reward)

    return objective_values, reward

  #print_output
  def logging(self, str):
    if self.print_output:
      print(str)
    with open(self.name() + ".txt", "a") as f:
      f.write(str + "\n")