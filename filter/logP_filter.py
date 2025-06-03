from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class LogPFilter(MolFilter):
    def __init__(self, max_logP=None, min_logP=None):
        self.max_logP = float("inf") if max_logP is None else max_logP
        self.min_logP = float("inf") if min_logP is None else min_logP

    #implement
    def mol_check(self, mol: Mol) -> bool:
        logP = Descriptors.MolLogP(mol)
        return self.min_logP <= logP and logP <= self.max_logP 