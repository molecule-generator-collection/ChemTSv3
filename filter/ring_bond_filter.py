from rdkit import Chem
from rdkit.Chem import Mol
from filter import MolFilter

class RingBondFilter(MolFilter):
    """
    Ref: https://github.com/jensengroup/GB_GA/tree/master by Jan H. Jensen 2018
    """
    # implement
    def mol_check(self, mol: Mol) -> int:
        if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
            return True

        ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))
        double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

        return not ring_allene and not double_bond_in_small_ring