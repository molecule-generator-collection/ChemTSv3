from rdkit import Chem
from rdkit.Chem import Mol
from chemtsv3.filter import MolValueFilter

class LinkerLengthFilter(MolValueFilter):
    """
    Filter for linker molecules.
    The filter excludes linkers whose maximum path length between the attachment points exceeds the user-defined value.
    """
    def __init__(self, allowed=None, disallowed=None, max=None, min=None):
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)

    def mol_value(self, mol: Mol) -> int:
        asterisk_list = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                asterisk_list.append(atom.GetIdx())
        if len(asterisk_list) != 2:
            return False
        dist_mat = Chem.GetDistanceMatrix(mol)
        return int(dist_mat[asterisk_list[0]][asterisk_list[1]])