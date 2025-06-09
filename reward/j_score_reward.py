import numpy as np
from rdkit.Chem import Descriptors
from reward import MolReward
from utils.third_party import sascorer

# edited from ChemTSv2
# ref: https://github.com/tsudalab/ChemTS/blob/4174c3600ebb47ed136b433b22a29c879824a6ba/mcts_logp_improved_version/add_node_type.py#L172

class JScoreReward(MolReward):
    def __init__(self, **kwargs):
        log_p_baseline = np.loadtxt("../data/misc/j_score/logP_values.txt")
        self.log_p_mean = np.mean(log_p_baseline)
        self.log_p_std = np.std(log_p_baseline)

        sa_baseline = np.loadtxt("../data/misc/j_score/SA_scores.txt")
        self.sa_mean = np.mean(sa_baseline)
        self.sa_std = np.std(sa_baseline)

        cs_baseline = np.loadtxt("../data/misc/j_score/cycle_scores.txt")
        self.cs_mean = np.mean(cs_baseline)
        self.cs_std = np.std(cs_baseline)
        super().__init__(**kwargs)
         
    #implement
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

    #implement
    def reward_from_objective_values(self, values):
        logP, sascore, ring_size_penalty = values
        logP_norm = (logP - self.log_p_mean) / self.log_p_std
        sascore_norm = (-sascore - self.sa_mean) / self.sa_std
        rs_penalty_norm = (-ring_size_penalty - self.cs_mean) / self.cs_std
        # jscore = logP - sascore - ring_size_penalty
        jscore = logP_norm + sascore_norm + rs_penalty_norm
        return jscore / (1 + abs(jscore))