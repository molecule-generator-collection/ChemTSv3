from rdkit.Chem import Mol, Descriptors
from filter import MolFilter
from node import MolNode

class ValidityFilter(MolFilter):
    def __init__(self, **kwargs):
        pass

    #implement
    def mol_check(self, mol: Mol) -> bool:
        if mol is None or mol.GetNumAtoms()==0:
            return False
        else:
            return True