# (monkey patch)
# import sys
# import types
# six_module = types.ModuleType("rdkit.six")
# six_module.iteritems = lambda d: d.items()
# sys.modules["rdkit.six"] = six_module 

from tdc import Oracle
from reward import SMILESReward

"""
mol-opt setting ref: https://github.com/wenhao-gao/mol_opt/blob/2da631be85af8d10a2bb43f2de76a03171166190/main/moldqn/environments/synth_env.py#L512
"""

class TDCReward(SMILESReward):
    is_single_objective = True    

    def __init__(self, objective: str):
        if type(objective) == str:
            self._name = objective.lower()
            self.oracle = Oracle(objective.upper())
        else:
            raise ValueError("Invalid objective (example of objective names: 'drd2', 'gsk3b', 'jnk3', or 'qed')")
        
    # implement
    def smiles_objective_functions(self):
        def raw_score(smiles):
            return self.oracle(smiles)

        return [raw_score]

    # implement
    def reward_from_objective_values(self, objective_values):
        score = objective_values[0]
        return score
    
    # override
    def name(self):
        return self._name