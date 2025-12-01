from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter

class LogPFilter(MolValueFilter):
    """
    `LogPFilter`: Excludes molecules whose LogP value falls outside the range [min, max]. (Default: (-∞, 5])
    """
    def __init__(self, max=5, min=None):
        super().__init__(max=max, min=min)    

    # implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.MolLogP(mol)