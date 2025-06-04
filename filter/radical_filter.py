from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class RadicalFilter(MolValueFilter):
    def __init__(self, allowed=0, **kwargs):
        super().__init__(allowed=allowed, **kwargs)
        
    #implement
    def mol_value(self, mol):
        return Descriptors.NumRadicalElectrons(mol)