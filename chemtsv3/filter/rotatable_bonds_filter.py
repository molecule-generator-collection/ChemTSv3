from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter

class RotatableBondsFilter(MolValueFilter):
    def __init__(self, max=10, min=None):
        """
        `RotatableBondsFilter`: Excludes molecules whose number of rotatable bonds falls outside the range [min, max]. (Default: [0, 10])
        """
        super().__init__(max, min)    

    # implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumRotatableBonds(mol)