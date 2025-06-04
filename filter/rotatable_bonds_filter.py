from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class RotatableBondsFilter(MolFilter):
    def __init__(self, max_rotatable_bonds=None, min_rotatable_bonds=None):
        self.max_rotatable_bonds = float("inf") if max_rotatable_bonds is None else max_rotatable_bonds
        self.min_rotatable_bonds = 0 if min_rotatable_bonds is None else min_rotatable_bonds

    #implement
    def mol_check(self, mol: Mol) -> bool:
        n = Descriptors.NumRotatableBonds(mol)
        return self.min_rotatable_bonds <= n and n <= self.max_rotatable_bonds