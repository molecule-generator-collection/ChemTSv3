import math
import numpy as np
from rdkit.Chem import Descriptors
from reward import MolReward

class LogPReward(MolReward):
  #implement
  @staticmethod
  def mol_objective_functions():
    def LogP(mol):
      if mol is None or mol.GetNumAtoms()==0:
        return float('nan')
      return Descriptors.MolLogP(mol)

    return [LogP]

  #implement
  @staticmethod
  def reward_from_objective_values(values, filtered_reward=-1):
    if math.isnan(values[0]):
      return filtered_reward
    return np.tanh(values[0] / 10)