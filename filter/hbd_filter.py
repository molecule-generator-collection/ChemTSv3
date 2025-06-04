from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class HBDFilter(MolFilter):
    def __init__(self, max_hydrogen_bond_donors=None, min_hydrogen_bond_donors=None):
        self.max_hydrogen_bond_donors = float("inf") if max_hydrogen_bond_donors is None else max_hydrogen_bond_donors
        self.min_hydrogen_bond_donors = 0 if min_hydrogen_bond_donors is None else min_hydrogen_bond_donors

    #implement
    def mol_check(self, mol: Mol) -> bool:
        n = Descriptors.NumHDonors(mol)
        return self.min_hydrogen_bond_donors <= n and n <= self.max_hydrogen_bond_donors