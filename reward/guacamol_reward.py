from guacamol.scoring_function import ScoringFunction
from rdkit import Chem
from reward import MolReward

class GuacaMolReward(MolReward):
    def __init__(self, scoring_function: ScoringFunction):    
        self.scoring_function = scoring_function
        
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