from rdkit import Chem
from rdkit.Chem import Mol
from filter import MolValueFilter

class ChargeFilter(MolValueFilter):
    def __init__(self, allowed=0, **kwargs):
        super().__init__(allowed=allowed, **kwargs)
        
    #implement
    def mol_value(self, mol: Mol) -> bool:
        return Chem.rdmolops.GetFormalCharge(mol)