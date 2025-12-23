import numpy as np
from rdkit.Chem import Descriptors
from chemtsv3.reward import MolReward

def sigmoid(x, a):
    return 1 / (1 + np.exp(-a * x))

class ExampleReward(MolReward):
    """Reward based on (LogP value - max ring size)."""
    def __init__(self, a):
        self.a = a
        
    def mol_objective_functions(self):
        """Return objective functions of the node; each function returns an objective value."""
        
        def log_p(mol):
            return Descriptors.MolLogP(mol)
        
        def max_ring_size(mol):
            ri = mol.GetRingInfo()
            max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
            return max_ring_size

        return [log_p, max_ring_size]
    
    def reward_from_objective_values(self, objective_values):
        """Compute the final reward based on the objective values calculated by objective_functions()."""
        log_p, max_ring_size = objective_values[0], objective_values[1]
        return sigmoid(x=log_p - max_ring_size, a=self.a) # It is recommended to scale the reward to the range [0, 1].