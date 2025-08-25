from rdkit import Chem
from rdkit.Chem import Mol, rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from reward import SingleMolReward

class SimilarityReward(SingleMolReward):    
    def __init__(self, target_smiles: str, radius: int=2, fp_size: int=2048):
        self.mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

        target = Chem.MolFromSmiles(target_smiles)
        self.target_fp = self.mfgen.GetFingerprint(target)
        
    # implement
    def mol_reward(self, mol: Mol) -> float:
        fp = self.mfgen.GetFingerprint(mol)
        return TanimotoSimilarity(fp, self.target_fp)