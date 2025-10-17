from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class HBAFilter(MolValueFilter):
    def __init__(self, max=10, min=None):
        super().__init__(max, min)    

    # implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumHAcceptors(mol)