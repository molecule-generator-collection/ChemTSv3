from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class LogPFilter(MolValueFilter):
    def __init__(self, max=5, min=None):
        super().__init__(max=max, min=min)    

    # implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.MolLogP(mol)