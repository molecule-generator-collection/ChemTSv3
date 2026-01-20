import pickle
import numpy as np
from rdkit import Chem

from chemtsv3.reward import MolReward
from chemtsv3.utils import calc_morgan_count, link_linker, max_gauss

class LinkerPermeabilityReward(MolReward):
    """
    Requires: tabpfn==2.1.3, scikit-learn==1.5.1
    """
    def __init__(self, model_path: str):
        with open(model_path, mode='rb') as f:
            self.permeability_model = pickle.load(f)
        
    def mol_objective_functions(self):
        def permeability(mol):
            Chem.SanitizeMol(mol)
            morganfp = calc_morgan_count(mol, r=2, dimension=500)
            X = np.array(morganfp, dtype=np.float32)
            y_pred = self.permeability_model.predict(X.reshape(1, -1))
            return y_pred[0]
        return [permeability]

    def reward_from_objective_values(self, objective_values):
        return max_gauss(objective_values[0], mu=0.25, sigma=1)