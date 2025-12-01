from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter

class HBDFilter(MolValueFilter):
    """
    Excludes molecules whose number of hydrogen bond donors falls outside the range [min, max]. (Default: [0, 5])
    """
    def __init__(self, max=5, min=None):
        super().__init__(max, min)

    # implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumHDonors(mol)