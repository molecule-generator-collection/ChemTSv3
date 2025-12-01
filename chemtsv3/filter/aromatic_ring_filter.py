from rdkit.Chem import Mol, Descriptors
from chemtsv3.filter import MolValueFilter

class AromaticRingFilter(MolValueFilter):
    """
    Excludes molecules whose number of aromatic rings falls outside the range [min, max]. (Default: [1, ∞))
    """
    def __init__(self, min=1, **kwargs):
        super().__init__(min=min, **kwargs)
    
    # implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumAromaticRings(mol)