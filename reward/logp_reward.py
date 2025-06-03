import math
import numpy as np
from rdkit.Chem import Descriptors
from reward import MolReward

class LogPReward(MolReward):        
    #implement
    def mol_objective_functions(self):
        def LogP(mol):
            # if mol is None or mol.GetNumAtoms()==0:
            #     return float('nan')
            return Descriptors.MolLogP(mol)

        return [LogP]

    #implement
    def reward_from_objective_values(self, values):
        # if math.isnan(values[0]):
        #     return self.filtered_reward
        return np.tanh(values[0] / 10)