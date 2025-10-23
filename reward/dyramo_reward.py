"""
ported and edited from: https://github.com/ycu-iil/DyRAMO/blob/main/reward/DyRAMO_reward.py
requires: lightgbm (tested: 3.3.5, 4.6.0)
"""
import json
import os
import pickle
import numpy as np
import lightgbm as lgb
from rdkit import DataStructs
from rdkit.Chem import AllChem
from reward import MolReward
from utils import max_gauss, min_gauss, rectangular

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

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
    def __init__(self, property: dict, ad: dict, exclude_approved: False):
        self.property = property
        self.ad = ad
        
        if exclude_approved:
            lgb_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dyramo/lgb_models_wo_approved_v1.json"))
            feature_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dyramo/fps_wo_approved_v1.pkl"))
        else:
            lgb_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dyramo/lgb_models.json"))
            feature_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dyramo/fps.pkl"))
            
        with open(lgb_models_path, mode="r") as l, \
            open(feature_path, mode="rb") as f:
            ms = json.load(l)
            self.lgb_models = {k: lgb.Booster(model_str=v) for k, v in ms.items()}
            self.feature_dict = pickle.load(f)
    
    # override
    def name(self):
        return "dyramo_reward"
    
    def mol_objective_functions(self):
        def egfr(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return self.lgb_models['EGFR'].predict(fp)[0]

        def stab(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return self.lgb_models['Stab'].predict(fp)[0] #Stab

        def perm(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return self.lgb_models['Perm'].predict(fp)[0]

        def egfr_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, self.feature_dict['EGFR'])
            num = self.ad['egfr']['num']
            return np.mean(similarity[:num])

        def stab_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, self.feature_dict['Stab'])
            num = self.ad['metabolic_stability']['num']
            return np.mean(similarity[:num])

        def perm_sim(mol):
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            similarity = calc_tanimoto_similarity(fp, self.feature_dict['Perm'])
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
    
