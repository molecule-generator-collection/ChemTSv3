import copy
from rdkit import Chem
from rdkit.Chem import Mol
from filter import MolFilter
from utils import mol_validity_check

class ValidityFilter(MolFilter):
    # implement
    def mol_check(self, mol: Mol) -> bool:
        return mol_validity_check(mol)