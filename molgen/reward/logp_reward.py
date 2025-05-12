from rdkit.Chem import Descriptors
import numpy as np
import math
from .reward import Reward

class LogP_reward(Reward):
  def get_objective_functions(conf):
    def LogP(node):
      mol = node.mol()
      if mol is None or mol.GetNumAtoms()==0:
        return float('nan')
      return Descriptors.MolLogP(node.mol())

    return [LogP]

  def reward_from_objective_values(values, conf):
    if math.isnan(values[0]):
      return conf["null_reward"]
    return np.tanh(values[0] / 10)