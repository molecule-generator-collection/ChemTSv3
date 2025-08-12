from rdkit.Chem import Mol
from filter import MolValueFilter

class HeavyAtomCountFilter(MolValueFilter):
    def __init__(self, max=45, min=None, allowed=None, disallowed=None):
        super().__init__(max=max, min=min, allowed=allowed, disallowed=disallowed)
        
    # implement
    def mol_value(self, mol: Mol) -> int:
        return mol.GetNumHeavyAtoms()