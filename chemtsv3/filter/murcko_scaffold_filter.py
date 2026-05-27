from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemtsv3.filter import MolFilter

class MurckoScaffoldFilter(MolFilter):
    def __init__(self, smiles: str, use_chirality: bool=False, generic: bool=True):
        """
        Excludes molecules whose Murcko scaffold does not contain the Murcko scaffold of the reference SMILES. If generic is set to True, atom and bond types are generalized before comparison.
        """
        ref_mol = Chem.MolFromSmiles(smiles)
        if ref_mol is None:
            raise ValueError(f"Invalid reference SMILES: {smiles}")

        ref_scaffold = MurckoScaffold.GetScaffoldForMol(ref_mol)
        if ref_scaffold is None or ref_scaffold.GetNumAtoms() == 0:
            raise ValueError(f"Reference molecule has no Murcko scaffold: {smiles}")

        if generic:
            ref_scaffold = MurckoScaffold.MakeScaffoldGeneric(ref_scaffold)

        self.smiles = smiles
        self.use_chirality = use_chirality
        self.generic = generic
        self.ref_scaffold = ref_scaffold

    # implement
    def mol_check(self, mol: Mol) -> bool:
        if mol is None:
            return False

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return False

        if self.generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

        return scaffold.HasSubstructMatch(self.ref_scaffold, useChirality=self.use_chirality)
