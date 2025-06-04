from rdkit import Chem
from rdkit.Chem import Mol
from filter import MolFilter

class ChargeFilter(MolFilter):
    def __init__(self, n=0):
        self.n = n
        
    #implement
    def mol_check(self, mol: Mol) -> bool:
        return Chem.rdmolops.GetFormalCharge(mol) == self.n