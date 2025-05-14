from math import sqrt
from typing import Any
from policy import Policy
from node import Node

class PUCT(Policy):
  def evaluate(node: Node, conf: dict[str, Any]=None):
    conf = conf or {}
    #c: exploration parameter, default: 1
    if node.n == 0 and conf.get("forced_rollout", True):
      return float("inf")
    u = conf.get("c", 1) * node.last_prob * sqrt(node.parent.n) / (1 + node.n)
    return node.mean_r + u