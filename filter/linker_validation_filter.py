from rdkit import Chem
from rdkit.Chem import Mol
from chemtsv3.filter import MolFilter
from chemtsv3.utils import add_atom_index_in_wildcard

class LinkerValidationFilter(MolFilter):
    """
    Filter for linker molecules.
    """
    def __init__(self, cores=None):
        self.cores = cores

    def mol_check(self, mol: Mol) -> bool:
        try:
            # Check if the molecule has a valid SMILES representation
            smi = Chem.MolToSmiles(mol)
        except:
            return False
        mol_ = Chem.MolFromSmiles(add_atom_index_in_wildcard(smi))
        rwmol = Chem.RWMol(mol_)
        cores_mol = [Chem.MolFromSmiles(s) for s in [self.cores['ligand_1'], self.cores['ligand_2']]]
        for m in cores_mol:
            rwmol.InsertMol(m)
        try:
            prod = Chem.molzip(rwmol)
            Chem.SanitizeMol(prod)
        except:
            return False
        return True