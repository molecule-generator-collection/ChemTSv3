from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter

class HBAFilter(MolValueFilter):
    """
    Excludes molecules whose number of hydrogen bond acceptors falls outside the range [min, max]. (Default: [0, 10])
    """
    def __init__(self, max=10, min=None):
        super().__init__(max, min)    

    # implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumHAcceptors(mol)