import scipy
import numpy as np
scipy.histogram = np.histogram # monkey patch
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.scoring_function import ScoringFunction
from guacamol.standard_benchmarks import amlodipine_rings, decoration_hop, hard_fexofenadine, hard_osimertinib, isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl, median_camphor_menthol, median_tadalafil_sildenafil, perindopril_rings, ranolazine_mpo, scaffold_hop, similarity, sitagliptin_replacement, valsartan_smarts, zaleplon_with_other_formula
from rdkit import Chem
from reward import MolReward

"""
mol-opt setting ref: https://github.com/wenhao-gao/mol_opt/blob/2da631be85af8d10a2bb43f2de76a03171166190/main/moldqn/environments/synth_env.py#L512
"""

class GuacaMolReward(MolReward):
    single_objective = True    

    def __init__(self, objective: ScoringFunction | GoalDirectedBenchmark | str):
        if type(objective) == GoalDirectedBenchmark:
            self.scoring_function = objective.wrapped_objective
        elif type(objective) == ScoringFunction:
            self.scoring_function = objective
        elif objective == "albuterol_similarity":
            self.scoring_function = similarity(smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', name='Albuterol',
                   fp_type='FCFP4', threshold=0.75).wrapped_objective
        elif objective == "amlodipine_mpo":
            self.scoring_function = amlodipine_rings().wrapped_objective
        elif objective == "celecoxib_rediscovery":
            self.scoring_function = similarity(smiles='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F', name='Celecoxib', fp_type='ECFP4', threshold=1.0, rediscovery=True).wrapped_objective
        elif objective == "deco_hop":
            self.scoring_function = decoration_hop().wrapped_objective
        elif objective == "fexofenadine_mpo":
            self.scoring_function = hard_fexofenadine().wrapped_objective
        elif objective == "isomers_c7h8n2o2":
            self.scoring_function = isomers_c7h8n2o2(mean_function='arithmetic').wrapped_objective
        elif objective == "isomers_c9h10n2o2pf2cl":
            self.scoring_function = isomers_c9h10n2o2pf2cl(mean_function='arithmetic', n_samples=100).wrapped_objective
        elif objective == "median1":
            self.scoring_function = median_camphor_menthol().wrapped_objective
        elif objective == "median2":
            self.scoring_function = median_tadalafil_sildenafil().wrapped_objective
        elif objective == "mestranol_similarity":
            self.scoring_function = similarity(smiles='COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1', name='Mestranol', fp_type='AP', threshold=0.75).wrapped_objective
        elif objective == "osimertinib_mpo":
            self.scoring_function = hard_osimertinib().wrapped_objective
        elif objective == "perindopril_mpo":
            self.scoring_function = perindopril_rings().wrapped_objective
        elif objective == "ranolazine_mpo":
            self.scoring_function = ranolazine_mpo().wrapped_objective
        elif objective == "scaffold_hop":
            self.scoring_function = scaffold_hop().wrapped_objective
        elif objective == "sitagliptin_mpo":
            self.scoring_function = sitagliptin_replacement().wrapped_objective
        elif objective == "thiothixene_rediscovery":
            self.scoring_function = similarity(smiles='CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1', name='Thiothixene', fp_type='ECFP4', threshold=1.0, rediscovery=True).wrapped_objective
        elif objective == "troglitazone_rediscovery":
            self.scoring_function = similarity(smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O', name='Troglitazone', fp_type='ECFP4', threshold=1.0, rediscovery=True).wrapped_objective
        elif objective == "valsartan_smarts":
            self.scoring_function = valsartan_smarts().wrapped_objective
        elif objective == "zaleplon_mpo":
            self.scoring_function = zaleplon_with_other_formula().wrapped_objective
        else:
            raise ValueError("Invalid objective.")
        
        if type(objective) == str:
            self._name = objective
        else:
            self._name = None
        
    # implement
    def mol_objective_functions(self):
        def raw_score(mol):
            smiles = Chem.MolToSmiles(mol)
            return self.scoring_function.score(smiles)

        return [raw_score]

    # implement
    def reward_from_objective_values(self, objective_values):
        score = objective_values[0]
        return score
    
    # override
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return super().name()