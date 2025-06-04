from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class AromaticRingFilter(MolFilter):
    #implement
    def mol_check(self, mol: Mol) -> bool:
        return Descriptors.NumAromaticRings(mol) > 0