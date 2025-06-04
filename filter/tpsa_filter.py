from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class TPSAFilter(MolFilter):
    def __init__(self, max_TPSA=140, min_TPSA=None):
        self.max_TPSA = float("inf") if max_TPSA is None else max_TPSA
        self.min_TPSA = -float("inf") if min_TPSA is None else min_TPSA

    #implement
    def mol_check(self, mol: Mol) -> bool:
        area = Descriptors.TPSA(mol)
        return self.min_TPSA <= area and area <= self.max_TPSA 