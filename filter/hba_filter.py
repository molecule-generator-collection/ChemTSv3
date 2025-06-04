from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class HBAFilter(MolFilter):
    def __init__(self, max_hydrogen_bond_acceptors=None, min_hydrogen_bond_acceptors=None):
        self.max_hydrogen_bond_acceptors = float("inf") if max_hydrogen_bond_acceptors is None else max_hydrogen_bond_acceptors
        self.min_hydrogen_bond_acceptors = 0 if min_hydrogen_bond_acceptors is None else min_hydrogen_bond_acceptors

    #implement
    def mol_check(self, mol: Mol) -> bool:
        n = Descriptors.NumHAcceptors(mol)
        return self.min_hydrogen_bond_acceptors <= n and n <= self.max_hydrogen_bond_acceptors