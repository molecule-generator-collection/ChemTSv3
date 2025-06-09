from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class LogPFilter(MolValueFilter):
    # implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.MolLogP(mol)