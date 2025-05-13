from math import sqrt
from .policy import Policy
from node.node import Node

class PUCT(Policy):
  def evaluate(node: Node, conf: dict):
    #c: exploration parameter, default: 1
    if node.parent is None:
      return node.mean_r
    u = conf.get("c", 1) * node.lastprob * sqrt(node.parent.n) / (1 + node.n)
    return node.mean_r + u