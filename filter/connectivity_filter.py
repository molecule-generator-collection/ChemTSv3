from rdkit.Chem import Mol, rdmolops
from filter import MolFilter

class ConnectivityFilter(MolFilter):
    # implement
    def mol_check(self, mol: Mol) -> bool:
        return len(rdmolops.GetMolFrags(mol)) == 1