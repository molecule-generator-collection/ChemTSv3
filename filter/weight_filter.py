from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class WeightFilter(MolValueFilter):
    #implement
    def mol_value(self, mol: Mol) -> float:
        return Descriptors.MolWt(mol)