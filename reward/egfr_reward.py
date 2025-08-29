"""
requires: lightgbm==3.2.1~3.3.5
"""
import pickle
import os
import numpy as np
from rdkit.Chem import AllChem
from reward import MolReward
from utils import max_gauss

LGB_MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/d_score/lgb_models.pickle"))

with open(LGB_MODELS_PATH, mode='rb') as models:
    lgb_models = pickle.load(models)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# python3.11/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.

def sigmoid(x, a):
    return 1 / (1 + np.exp(-a * x))

class EGFRReward(MolReward):
    is_single_objective = True
    
    def __init__(self, type="max_gauss", a: float=0.2, alpha: float=1, mu: float=9, sigma: float=2):
        if type == "max_gauss":
            self.type = "max_gauss"
            self.alpha = alpha
            self.mu = mu
            self.sigma = sigma
        else:
            self.type = "sigmoid"
            self.a = a
        
    # implement
    def mol_objective_functions(self):
        def egfr(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["EGFR"].predict(fp)[0]

        return [egfr]

    # implement
    def reward_from_objective_values(self, objective_values):
        raw_egfr = objective_values[0]
        if raw_egfr is None:
            return -1
        
        if self.type == "max_gauss":
            scaled_egfr = max_gauss(raw_egfr, self.alpha, self.mu, self.sigma)
        else:
            scaled_egfr = sigmoid(raw_egfr, self.a)

        return scaled_egfr