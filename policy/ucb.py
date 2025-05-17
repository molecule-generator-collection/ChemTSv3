from math import log, sqrt
from typing import Any
from policy import Policy
from node import Node

class UCB(Policy):
  #implement
  def evaluate(node: Node, c=1):
    #c: exploration parameter
    if node.n == 0:
      return float("inf")
    u = c * sqrt(2 * log(node.parent.n) / (node.n))
    return node.mean_r + u