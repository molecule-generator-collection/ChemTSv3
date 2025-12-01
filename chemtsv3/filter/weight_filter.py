from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter

class WeightFilter(MolValueFilter):
    def __init__(self, max=500, min=None):
        """
        Excludes molecules whose molecular weight falls outside the range [min, max]. (Default: [0, 500])
        """
        super().__init__(max, min)

    # implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.MolWt(mol)