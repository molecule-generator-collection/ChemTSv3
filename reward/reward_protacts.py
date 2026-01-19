import pickle
import numpy as np
from rdkit import Chem

from reward import MolReward
from utils import max_gauss
from utils.linker_utils import calc_MorganCount, link_linker

class LinkerpermeabilityReward(MolReward):
    def __init__(self, cores: dict, model_file: str):
        self.cores = cores
        with open(model_file, mode='rb') as f:
            self.permeability_model = pickle.load(f)
        
    def mol_objective_functions(self):
        def Permeability(mol):
            prod = link_linker(self.cores, mol)
            Chem.SanitizeMol(prod)
            morganfp = calc_MorganCount(prod, r=2, dimension=500)
            X = np.array(morganfp, dtype=np.float32)
            y_pred = self.permeability_model.predict(X.reshape(1, -1))
            return y_pred[0]
        return [Permeability]

    def reward_from_objective_values(self, objective_values):
        return max_gauss(objective_values[0], mu=0.25, sigma=1)
