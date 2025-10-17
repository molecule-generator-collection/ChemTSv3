from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter
from utils.third_party import sascorer

class SAScoreFilter(MolValueFilter):
    def __init__(self, max=3.5, min=None):
        super().__init__(max=max, min=min)

    # implement
    def mol_value(self, mol: Mol) -> float:
        return sascorer.calculateScore(mol)