from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Mol
from rdkit import DataStructs

from filter import MolValueFilter

class LinkerSimilarityFilter(MolValueFilter):
    def __init__(self, allowed=None, disallowed=None, max=None, min=None, linker_file=None, fpSize=None, radius=None):
        super().__init__(allowed=allowed, disallowed=disallowed, max=max, min=min)
        with open(linker_file, mode='r') as f:
            linkersim_protac_linker_list = f.readlines()
        linkersim_protac_linker_mol_list = [Chem.MolFromSmiles(i) for i in linkersim_protac_linker_list]
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
        self.linkersim_linker_morgan_list = [self.generator.GetFingerprint(i) for i in linkersim_protac_linker_mol_list]

    def mol_value(self, mol: Mol) -> int:
        ref_morgan_fp = self.generator.GetFingerprint(mol)
        morgan_fp_tanimoto_list = DataStructs.BulkTanimotoSimilarity(ref_morgan_fp, self.linkersim_linker_morgan_list)
        return max(morgan_fp_tanimoto_list)
    
    def mol_value_linker(self, mol: Mol, _link_mol: Mol) -> int:
        return self.mol_value(mol)