import numpy as np
from rdkit.Chem import Descriptors
from reward import MolReward
from utils.third_party import sascorer

"""
ported from ChemTSv2: https://github.com/molecule-generator-collection/ChemTSv2/blob/master/reward/Jscore_reward.py
ref: https://github.com/tsudalab/ChemTS/blob/4174c3600ebb47ed136b433b22a29c879824a6ba/mcts_logp_improved_version/add_node_type.py#L172
"""

LOG_P_BASELINE = np.loadtxt("../data/j_score/logP_values.txt")
LOG_P_MEAN = np.mean(LOG_P_BASELINE)
LOG_P_STD = np.std(LOG_P_BASELINE)

SA_BASELINE = np.loadtxt("../data/j_score/SA_scores.txt")
SA_MEAN = np.mean(SA_BASELINE)
SA_STD = np.std(SA_BASELINE)

CA_BASELINE = np.loadtxt("../data/j_score/cycle_scores.txt")
CS_MEAN = np.mean(CA_BASELINE)
CS_STD = np.std(CA_BASELINE)

class JScoreReward(MolReward):
    # implement
    def mol_objective_functions(self):
        def log_p(mol):
            return Descriptors.MolLogP(mol)

        def sa_score(mol):
            return sascorer.calculateScore(mol)

        def ring_size_penalty(mol):
            ri = mol.GetRingInfo()
            max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
            return max_ring_size - 6

        return [log_p, sa_score, ring_size_penalty]

    # implement
    def reward_from_objective_values(self, objective_values):
        logP, sascore, ring_size_penalty = objective_values
        logP_norm = (logP - LOG_P_MEAN) / LOG_P_STD
        sascore_norm = (-sascore - SA_MEAN) / SA_STD
        rs_penalty_norm = (-ring_size_penalty - CS_MEAN) / CS_STD
        # jscore = logP - sascore - ring_size_penalty
        jscore = logP_norm + sascore_norm + rs_penalty_norm
        return jscore / (1 + abs(jscore))