from rdkit.Chem import Mol, rdmolops
from chemtsv3.filter import MolValueFilter

class ConnectivityFilter(MolValueFilter):
    """
    Excludes molecules whose number of disconnected fragments is not 1.
    """
    def __init__(self, allowed=1, disallowed=None, max=None, min=None):
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)
    
    # implement
    def mol_value(self, mol: Mol) -> bool:
        return len(rdmolops.GetMolFrags(mol))