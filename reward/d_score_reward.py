"""
edited from ChemTSv2: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/dscore_reward.py
requires: lightgbm==3.2.1~3.3.5
recommended: conda install -c conda-forge lightgbm=3.3.5
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from reward import MolReward
from utils import max_gauss, min_gauss, rectangular
from utils.third_party import sascorer

LGB_MODELS_PATH = "../data/d_score/lgb_models.pickle"
SURE_CHEMBL_ALERTS_PATH = "../data/d_score/sure_chembl_alerts.txt"
CHEMBL_FPS_PATH = "../data/d_score/chembl_fps.npy"

with open(LGB_MODELS_PATH, mode='rb') as models,\
    open(SURE_CHEMBL_ALERTS_PATH, mode='rb') as alerts, \
    open(CHEMBL_FPS_PATH, mode='rb') as fps:
    lgb_models = pickle.load(models)
    smarts = pd.read_csv(alerts, header=None, sep='\t')[1].tolist()
    alert_mols = [Chem.MolFromSmarts(smart) for smart in smarts if Chem.MolFromSmarts(smart) is not None]
    chebml_fps = np.load(fps, allow_pickle=True).item()


def scale_objective_value(params, value):
    scaling = params["type"]
    if scaling == "max_gauss":
        return max_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "min_gauss":
        return min_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling == "minmax":
        return (value - params["min"]) / (params["max"] - params["min"])
    elif scaling == "rectangular":
        return rectangular(value, params["min"], params["max"])
    elif scaling == "identity":
        return value
    else:
        raise ValueError("Set the scaling function from one of 'max_gauss', 'min_gauss', 'minimax', rectangular, or 'identity'")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# python3.11/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.

class DScoreReward(MolReward):
    def __init__(self, params: dict[str, dict]):
        self.params = params
        
    # implement
    def mol_objective_functions(self):
        def egfr(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["EGFR"].predict(fp)[0]

        def erbb2(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["ERBB2"].predict(fp)[0]

        def abl(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["ABL"].predict(fp)[0]

        def src(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["SRC"].predict(fp)[0]

        def lck(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["LCK"].predict(fp)[0]

        def pdgfr_beta(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["PDGFRbeta"].predict(fp)[0]

        def vegfr2(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["VEGFR2"].predict(fp)[0]

        def fgfr1(mol):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["FGFR1"].predict(fp)[0]

        def ephb4(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["EPHB4"].predict(fp)[0]

        def solubility(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Sol"].predict(fp)[0]

        def permeability(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Perm"].predict(fp)[0]

        def metabolic_stability(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Meta"].predict(fp)[0]

        def toxicity(mol):
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models["Tox"].predict(fp)[0]

        def sa_score(mol):
            return sascorer.calculateScore(mol)

        def qed(mol):
            try:
                return Chem.QED.qed(mol)
            except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
                return None

        # taken from　https://github.com/jrwnter/mso.
        def molecular_weight(mol):
            """molecular weight"""
            mw = Chem.Descriptors.MolWt(mol)
            return mw

        # taken from　https://github.com/jrwnter/mso.
        def tox_alert(mol):
            """
            0 if a molecule matches a structural alert as defined by the included list from surechembl.
            """
            if np.any([mol.HasSubstructMatch(alert) for alert in alert_mols]):
                score = 0
            else:
                score = 1
            return score

        # taken from　https://github.com/jrwnter/mso.
        def has_chembl_substruct(mol):
            """0 for molecuels with substructures (ECFP2 that occur less often than 5 times in ChEMBL."""
            fp_query = AllChem.GetMorganFingerprint(mol, 1, useCounts=False)
            if np.any([bit not in chebml_fps for bit in fp_query.GetNonzeroElements().keys()]):
                return 0
            else:
                return 1

        return [egfr, erbb2, abl, src, lck, pdgfr_beta, vegfr2, fgfr1, ephb4, solubility, permeability, metabolic_stability,
                toxicity, sa_score, qed, molecular_weight, tox_alert, has_chembl_substruct]

    # implement
    def reward_from_objective_values(self, values):
        if None in values:
            return -1

        objectives = [f.__name__ for f in self.mol_objective_functions()]

        scaled_values = []
        weights = []
        for objective, value in zip(objectives, values):
            if objective == "SAscore":
                # SAscore is made negative when scaling because a smaller value is more desirable.
                scaled_values.append(scale_objective_value(self.params[objective], -1 * value))
            else:
                scaled_values.append(scale_objective_value(self.params[objective], value))
            weights.append(self.params[objective]["weight"])

        multiplication_value = 1
        for v, w in zip(scaled_values, weights):
            multiplication_value *= v**w
        dscore = multiplication_value ** (1/sum(weights))

        return dscore