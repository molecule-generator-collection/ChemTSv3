import numpy as np
from rdkit.Chem import Descriptors
from chemtsv3.reward import BatchReward

class ExampleBatchReward(BatchReward):
    def __init__(self, n_batch):
        self._n_batch = n_batch
        
    def n_batch(self):
        return self._n_batch
    
    def objective_values_and_rewards(self, nodes):
        results = [] 
        for n in nodes:
            mol = n.mol()
            logp = Descriptors.MolLogP(mol)
            reward = np.tanh(logp / 10)
            results.append([[logp], reward])
        return results

    def objective_names(self):
        return ["logp"]
