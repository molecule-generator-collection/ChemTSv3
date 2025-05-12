from math import log, sqrt
from .policy import Policy
from node.node import Node

class UCB(Policy):
  def evaluate(node: Node, conf={"c":1}):
    if node.parent is None:
      return node.mean_r
    if node.n == 0:
      return float("inf")
    u = conf["c"] * sqrt(2 * log(node.parent.n) / (node.n))
    return node.mean_r + u