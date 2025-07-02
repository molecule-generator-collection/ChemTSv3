from rdkit.Chem import Descriptors
from filter import MolValueFilter

class RadicalFilter(MolValueFilter):
    def __init__(self, allowed=0, **kwargs):
        super().__init__(allowed=allowed, **kwargs)
        
    # implement
    def mol_value(self, mol) -> int:
        return Descriptors.NumRadicalElectrons(mol)