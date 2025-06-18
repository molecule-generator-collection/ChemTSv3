import copy
from rdkit import Chem
from rdkit.Chem import Mol
from filter import MolFilter

class ValidityFilter(MolFilter):  
    # implement
    def mol_check(self, mol: Mol) -> bool:
        if mol is None or mol.GetNumAtoms()==0:
            return False
        _mol = copy.deepcopy(mol)
        if Chem.SanitizeMol(_mol, catchErrors=True).name != "SANITIZE_NONE":
            return False
        return True