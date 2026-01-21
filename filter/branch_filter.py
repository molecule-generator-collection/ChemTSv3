from rdkit import Chem
from rdkit.Chem import Mol
from chemtsv3.filter import MolValueFilter

class BranchFilter(MolValueFilter):
    """
    Filter for linker molecules.
    The filter excludes linkers whose shortest path between the two attachment points contains at least the user-defined number of branching atoms, including those in ring structures.
    """
    def __init__(self, allowed=None, disallowed=None, max=None, min=None):
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)

    def mol_value(self, mol: Mol) -> int:
        try:
            smi = Chem.MolToSmiles(mol)
        except:
            return None
        if smi.count("*") != 2:
            return None
        asterisk_list = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                asterisk_list.append(atom.GetIdx())
        dist_mat = Chem.GetDistanceMatrix(mol)
        ref_len = dist_mat[asterisk_list[0]][asterisk_list[1]]
        most_short_len_atom = []
        branch_counter = 0
        for i in range(len(dist_mat[asterisk_list[0]])):
            a = dist_mat[asterisk_list[0]][i] + dist_mat[asterisk_list[1]][i]
            if a != ref_len:
                continue
            else:
                most_short_len_atom.append(i)
        if mol.GetRingInfo().NumRings() == 0:
            ring_atom = []
        else:
            ring_atom = list(mol.GetRingInfo().AtomRings()[0])
            ring_atom = [i for i in ring_atom if i not in most_short_len_atom]
        ref_atom = most_short_len_atom + ring_atom
        all_atom = [atom.GetIdx() for atom in mol.GetAtoms()]
        branch_atom = [i for i in all_atom if i not in ref_atom]
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in branch_atom:
                continue
            branch_counter += 1
        return branch_counter