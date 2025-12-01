from rdkit.Chem import Mol
from chemtsv3.filter import MolValueFilter

class AtomCountFilter(MolValueFilter):
    def __init__(self, max=60, min=6, allowed=None, disallowed=None):
        super().__init__(max=max, min=min, allowed=allowed, disallowed=disallowed)
        
    # implement
    def mol_value(self, mol: Mol) -> int:
        return mol.GetNumAtoms()