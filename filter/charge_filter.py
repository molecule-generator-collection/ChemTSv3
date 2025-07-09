from rdkit import Chem
from rdkit.Chem import Mol
from filter import MolValueFilter

class ChargeFilter(MolValueFilter):
    def __init__(self, allowed=0, disallowed=None, max=None, min=None):
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)
        
    # implement
    def mol_value(self, mol: Mol) -> int:
        return Chem.rdmolops.GetFormalCharge(mol)