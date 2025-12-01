from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter

class TPSAFilter(MolValueFilter):
    """
    Excludes molecules whose topological polar surface area (TPSA) falls outside the range [min, max]. (Default: [0, 140])
    """
    def __init__(self, max=140, min=None):
        super().__init__(max=max, min=min)

    # implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.TPSA(mol)