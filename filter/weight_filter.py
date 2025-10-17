from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class WeightFilter(MolValueFilter):
    def __init__(self, max=500, min=None):
        super().__init__(max, min)

    # implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.MolWt(mol)