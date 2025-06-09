from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class TPSAFilter(MolValueFilter):
    def __init__(self, max=140, **kwargs):
        super().__init__(max=max, **kwargs)

    # implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.TPSA(mol)