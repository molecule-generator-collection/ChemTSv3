from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class AromaticRingFilter(MolValueFilter):
    def __init__(self, min=1, **kwargs):
        super().__init__(min=min, **kwargs)
    
    #implement
    def mol_value(self, mol: Mol) -> int:
        return Descriptors.NumAromaticRings(mol)