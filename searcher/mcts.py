import time, datetime
from typing import Type
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from node.node import Node
from edgepredictor.edgepredictor import EdgePredictor
from policy.policy import Policy
from reward.reward import Reward
from reward.logp_reward import LogP_reward

class MCTS():
  def __init__(self, edgepredictor: EdgePredictor, reward: Type[Reward] = LogP_reward, reward_conf: dict = {"null_reward": -1},rollout_limit=4096, print_output=True, verbose=False, name=None):
    self.edgepredictor = edgepredictor
    self.reward = reward
    self.reward_conf = reward_conf #specify at least all of: "null_reward"
    self.rollout_limit = rollout_limit #max sentence length, unused
    self.record: dict[str, tuple[list[float], float]] = {}
    self.unique_molkeys = []
    self.rewards = []
    self.n_rollouts = []
    self.times = []
    self.print_output = print_output
    self.verbose = verbose
    self.name = name
    self.filename = self.name
    self.count_rollouts = 0
    self.passed_time = 0

  #for expansion_threshold
  @staticmethod
  def select_indices_by_threshold(probs: list[float], expansion_threshold: float) -> list[int]:
      probs = np.array(probs)
      sorted_indices = np.argsort(-probs)
      sorted_probs = probs[sorted_indices]
      cumulative_probs = np.cumsum(sorted_probs)
      cutoff = np.searchsorted(cumulative_probs, expansion_threshold)
      return sorted_indices[:cutoff + 1].tolist()

  #print_output
  def logging(self, str):
    if self.print_output:
      print(str)
    with open(self.filename + ".txt", "a") as f:
      f.write(str + "\n")

  #visualize results
  def plot(self, type="reward_call", cutoff=None, maxline=False):
    if type == "time":
      x, y = self.times, self.rewards
    elif type == "num_rollout":
      self.n_rollouts, self.rewards
    else:
      x, y = list(range(1, len(self.rewards)+1)), self.rewards

    if cutoff != None:
      x, y = x[:cutoff], y[:cutoff]

    plt.clf()
    plt.scatter(x, y, s=1)
    #plt.title("model: " + model_name + ", policy: " + policy + ", c = " + str(c))
    plt.title(self.name)

    if type == "time":
      plt.xlim(0,x[-1])
      plt.xlabel("seconds_passed")
    elif type == "num_rollouts":
      plt.xlim(0,x[-1])
      plt.xlabel("num_rollouts")
    else: #"reward call"
      plt.xlim(0,len(x))
      plt.xlabel("reward calls (unique valid helms)")

    plt.ylim(-1,1)
    plt.ylabel("reward")
    plt.grid(axis="y")

    if maxline:
      max(y)
      y_max = np.max(y)
      plt.axhline(y=y_max, color='red', linestyle='--', label=f'y={y_max:.5f}')

    plt.legend()
    plt.show()

  def log_unique_mol(self, key, objective_values, reward):
      self.logging(str(len(self.unique_molkeys)) + "- time: " +  "{:.2f}".format(self.passed_time) + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(reward) + ", mol: " + key)
      self.unique_molkeys.append(key)
      self.rewards.append(reward)
      self.n_rollouts.append(self.count_rollouts)
      self.times.append(time)

  def grab_objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
    key = str(node)
    if key in self.record:
      if self.verbose:
        self.logging("already in dict: " + key + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(self.record[key][1]))
      return self.record[key]
    objective_values, reward = self.reward.objective_values_and_reward(node, conf=self.reward_conf)
    self.record[key] = (objective_values, reward)

    if reward != self.reward_conf["null_reward"]:
      self.log_unique_mol(key, objective_values, reward)
    else:
      if self.verbose:
        self.logging("invalid mol: " + key)

    return objective_values, reward

  def _expand(self, node: Node, expansion_threshold=0.995):
    if node.is_terminal():
      return

    #apply expansion_threshold
    nodes = self.edgepredictor.nextnodes_with_probs(node)
    probs = [node.lastprob for node in nodes]
    remaining_ids = MCTS.select_indices_by_threshold(probs, expansion_threshold)

    for id in remaining_ids:
      node.children[id] = nodes[id]

  def _eval(self, node: Node, expansion_threshold=0.995):
    if node.is_terminal():
      objective_values, reward = self.grab_objective_values_and_reward(node)
      node.sum_r = node.mean_r = -float("inf")
      return reward
    if not bool(node.children): #if empty
      self._expand(node, expansion_threshold=expansion_threshold)
    objective_values, reward = self._rollout(node)
    return reward

  def _rollout(self, node):
    if node.idtensor.numel() >= self.rollout_limit:
      return self.reward_conf["null_reward"]
    mol = self.edgepredictor.randomgen(node)
    self.count_rollouts += 1
    return self.grab_objective_values_and_reward(mol)

  def _backpropagate(self, node, value):
    while node:
      node.n += 1
      node.sum_r += value
      node.mean_r = node.sum_r / node.n
      node = node.parent

  def search(self, root: Node, policy: Type[Policy], policy_conf={"c":1}, expansion_threshold=0.995, exhaust_backpropagate=False, max_rollouts=None, time_limit=None, max_generations=None):
    #exhaust_backpropagate: whether to backpropagate or not when every terminal node under the node is already explored (only once: won't be visited again)
    #expansion_threshold: [0-1], ignore children with low transition probabilities based on this value
    assert (max_rollouts is not None) or (time_limit is not None) or (max_generations is not None), \
        "specify at least one of num_genrations, max_rollouts or time_limit"

    if self.name is None:
      self.filename = self.name = str(datetime.datetime.now())
      #old: self.name = policy + "_c" + '{:.2f}'.format(c) + "_" + str(datetime.datetime.now())

    time_start = time.time()
    initial_time = self.passed_time
    initial_count_rollouts = self.count_rollouts
    initial_count_generations = len(self.unique_molkeys)

    self._expand(root, expansion_threshold=expansion_threshold)

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
        node = max(node.children.values(), key=lambda n: policy.evaluate(n))
        if node.sum_r == -float("inf"): #already exhausted every terminal under this
          if self.verbose:
            self.logging("!------exhaust every terminal under: " + str(node.parent) + "------!")
          if exhaust_backpropagate:
            value = self._eval(node, expansion_threshold=expansion_threshold)
            self._backpropagate(node, value)
          node.parent.sum_r = node.parent.mean_r = -float("inf")
          node = root
          continue
      value = self._eval(node, expansion_threshold=expansion_threshold)
      self._backpropagate(node, value)

    print("Search is completed.")