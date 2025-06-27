import math
import numpy as np
from rdkit.Chem import Descriptors
from reward import MolReward

class LogPReward(MolReward):        
    # implement
    def mol_objective_functions(self):
        def log_p(mol):
            return Descriptors.MolLogP(mol)

        return [log_p]

    # implement
    def reward_from_objective_values(self, objective_values):
        return np.tanh(objective_values[0] / 10)