from rdkit.Chem import Descriptors
from filter import MolValueFilter

class RadicalFilter(MolValueFilter):
    def __init__(self, allowed=0, disallowed=None, max=None, min=None):
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)
        
    # implement
    def mol_value(self, mol) -> int:
        return Descriptors.NumRadicalElectrons(mol)