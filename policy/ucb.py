from math import log, sqrt
from typing import Any
from policy import Policy
from node import Node

class UCB(Policy):
  def evaluate(node: Node, conf: dict[str, Any] = None):
    conf = conf or {}
    #c: exploration parameter, default: 1
    if node.n == 0:
      return float("inf")
    u = conf.get("c", 1) * sqrt(2 * log(node.parent.n) / (node.n))
    return node.mean_r + u