"""
ported and edited from: https://github.com/ycu-iil/DyRAMO/blob/main/reward/DyRAMO_reward.py
requires: lightgbm==3.2.1~3.3.5
"""
import os
import pickle
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from reward import MolReward
from utils import max_gauss, min_gauss, rectangular

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

LGB_MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dyramo/lgb_models.pkl"))
FEATURE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dyramo/fps.pkl"))

with open(LGB_MODELS_PATH, mode='rb') as l, \
    open(FEATURE_PATH, mode='rb') as f:
    lgb_models = pickle.load(l)
    feature_dict = pickle.load(f)

def step(x, threshold):
    if x >= threshold:
        return 1
    else:
        return 0

def scale_objective_value(params, value):
    scaling = params['scaler']
    if scaling == 'max_gauss':
        return max_gauss(value, 1.0, params['mu'], params['sigma'])
    elif scaling == 'min_gauss':
        return min_gauss(value, 1.0, params['mu'], params['sigma'])
    elif scaling == 'step':
        return step(value, params['threshold'])
    elif scaling == "rectangular":
        return rectangular(value, params["min"], params["max"])
    elif scaling == 'identity':
        return value
    else:
        raise ValueError("Set the scaling function from one of 'max_gauss', 'min_gauss', 'rectangular', 'inverted_step' or 'identity'")

def calc_tanimoto_similarity(feat_generated, feat_train):
    similarity = DataStructs.BulkTanimotoSimilarity(feat_generated, feat_train, returnDistance=False)
    similarity_sorted = sorted(similarity, reverse=True) # Sort by similarity in descending order
    return similarity_sorted

class DyRAMOReward(MolReward):
    def __init__(self, property: dict, ad: dict):
        self.property = property
        self.ad = ad
    
    # override
    def name(self):
        return "dyramo_reward"
    
    def mol_objective_functions(self):
        def egfr(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['EGFR'].predict(fp)[0]

        def stab(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Stab'].predict(fp)[0] #Stab

        def perm(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models['Perm'].predict(fp)[0]

        def egfr_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['EGFR'])
            num = self.ad['egfr']['num']
            return np.mean(similarity[:num])

        def stab_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['Stab'])
            num = self.ad['metabolic_stability']['num']
            return np.mean(similarity[:num])

        def perm_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, feature_dict['Perm'])
            num = self.ad['permeability']['num']
            return np.mean(similarity[:num])

        return [egfr, stab, perm, egfr_sim, stab_sim, perm_sim]
    
    def reward_from_objective_values(self, objective_values):
        if None in objective_values:
            return -1

        egfr, stab, perm, egfr_sim, stab_sim, perm_sim = objective_values

        # AD filter
        is_in_AD = [] # 0: out of AD, 1: in AD
        is_in_AD.append(scale_objective_value(self.ad['egfr'], egfr_sim))
        is_in_AD.append(scale_objective_value(self.ad['metabolic_stability'], stab_sim))
        is_in_AD.append(scale_objective_value(self.ad['permeability'], perm_sim))
        weights_AD = []
        weights_AD.append(self.ad['egfr']['weight'])
        weights_AD.append(self.ad['metabolic_stability']['weight'])
        weights_AD.append(self.ad['permeability']['weight'])
        for i, w in zip(is_in_AD, weights_AD):
            if w == 0:
                continue
            if i == 0:
                return 0

        # Property
        scaled_values = []
        scaled_values.append(scale_objective_value(self.property['egfr'], egfr))
        scaled_values.append(scale_objective_value(self.property['metabolic_stability'], stab))
        scaled_values.append(scale_objective_value(self.property['permeability'], perm))
        weights = []
        weights.append(self.property['egfr']['weight'])
        weights.append(self.property['metabolic_stability']['weight'])
        weights.append(self.property['permeability']['weight'])

        multiplication_value = 1
        for v, w in zip(scaled_values, weights):
            multiplication_value *= v**w
        dscore = multiplication_value ** (1/sum(weights))

        return dscore
    
