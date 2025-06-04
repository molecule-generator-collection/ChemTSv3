from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class RotatableBondsFilter(MolValueFilter):
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumRotatableBonds(mol)