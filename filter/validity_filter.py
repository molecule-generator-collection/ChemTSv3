from rdkit.Chem import Mol
from filter import MolFilter

class ValidityFilter(MolFilter):
    #implement
    def mol_check(self, mol: Mol) -> bool:
        if mol is None or mol.GetNumAtoms()==0:
            return False
        else:
            return True