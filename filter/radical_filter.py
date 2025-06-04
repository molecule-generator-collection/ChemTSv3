from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class RadicalFilter(MolFilter):
    #implement
    def mol_check(self, mol: Mol) -> bool:
        return Descriptors.NumRadicalElectrons(mol) == 0