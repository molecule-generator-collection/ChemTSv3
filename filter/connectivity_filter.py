from rdkit.Chem import Mol, rdmolops
from filter import MolValueFilter

class ConnectivityFilter(MolValueFilter):
    def __init__(self, allowed=1, disallowed=None, max=None, min=None):
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)
    
    # implement
    def mol_value(self, mol: Mol) -> bool:
        return len(rdmolops.GetMolFrags(mol))