import copy
from rdkit import Chem
from rdkit.Chem import Mol
from chemtsv3.filter import MolFilter
from chemtsv3.utils import mol_validity_check

class ValidityFilter(MolFilter):
    """
    Excludes invalid molecule objects. Since other filters and rewards typically assume validity and do not recheck it, this filter should usually be applied first in molecular generation.
    """
    # implement
    def mol_check(self, mol: Mol) -> bool:
        return mol_validity_check(mol)