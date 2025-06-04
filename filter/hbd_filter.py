from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class HBDFilter(MolValueFilter):
    #implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumHDonors(mol)