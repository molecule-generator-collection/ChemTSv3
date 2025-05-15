from rdkit.Chem import Descriptors
import numpy as np
import math
from typing import Any
from reward import MolReward

class LogPReward(MolReward):
  #override
  @staticmethod
  def mol_objective_functions():
    def LogP(mol):
      if mol is None or mol.GetNumAtoms()==0:
        return float('nan')
      return Descriptors.MolLogP(mol)

    return [LogP]

  #override
  @staticmethod
  def reward_from_objective_values(values, filtered_reward=-1):
    if math.isnan(values[0]):
      return filtered_reward
    return np.tanh(values[0] / 10)