from rdkit.Chem import Descriptors, Mol
from chemtsv3.filter import MolValueFilter

class RadicalFilter(MolValueFilter):
    def __init__(self, allowed=0, disallowed=None, max=None, min=None):
        """
        Excludes molecules whose number of radical electrons is not 0.
        """
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)
        
    # implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumRadicalElectrons(mol)