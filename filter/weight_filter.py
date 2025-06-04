from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class WeightFilter(MolFilter):
    def __init__(self, max_mol_weight=None, min_mol_weight=None):
        self.max_mol_weight = float("inf") if max_mol_weight is None else max_mol_weight
        self.min_mol_weight = -float("inf") if min_mol_weight is None else min_mol_weight

    #implement
    def mol_check(self, mol: Mol) -> bool:
        w = Descriptors.MolWt(mol)
        return self.min_mol_weight <= w and w <= self.max_mol_weight 