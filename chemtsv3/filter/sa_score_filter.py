from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter
from chemtsv3.utils.third_party import sascorer

class SAScoreFilter(MolValueFilter):
    """
    Excludes molecules whose synthetic accessibility score (SA Score) falls outside the range [min, max]. (Default: [1, 3.5])
    """
    def __init__(self, max=3.5, min=None):
        super().__init__(max=max, min=min)

    # implement
    def mol_value(self, mol: Mol) -> float:
        return sascorer.calculateScore(mol)