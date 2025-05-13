from rdkit.Chem import Descriptors
import numpy as np
import math
from typing import Any
from reward import MolReward

class LogP_reward(MolReward):
  #override
  @staticmethod
  def mol_objective_functions(conf: dict[str, Any] = None):
    def LogP(mol):
      if mol is None or mol.GetNumAtoms()==0:
        return float('nan')
      return Descriptors.MolLogP(mol)

    return [LogP]

  #override
  @staticmethod
  def reward_from_objective_values(values, conf: dict[str, Any] = None):
    conf = conf or {}
    if math.isnan(values[0]):
      return conf.get("null_reward", -1)
    return np.tanh(values[0] / 10)