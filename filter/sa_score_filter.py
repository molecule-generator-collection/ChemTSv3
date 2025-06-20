from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter
from utils.third_party import sascorer

class SAScoreFilter(MolValueFilter):
    # implement
    def mol_value(self, mol: Mol) -> float:
        return sascorer.calculateScore(mol)