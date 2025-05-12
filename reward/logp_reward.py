from rdkit.Chem import Descriptors
import numpy as np
import math
from .reward import MolReward

class LogP_reward(MolReward):
  def mol_objective_functions(conf):
    def LogP(mol):
      if mol is None or mol.GetNumAtoms()==0:
        return float('nan')
      return Descriptors.MolLogP(mol)

    return [LogP]

  def reward_from_objective_values(values, conf):
    if math.isnan(values[0]):
      return conf["null_reward"]
    return np.tanh(values[0] / 10)