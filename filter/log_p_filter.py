from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class LogPFilter(MolFilter):
    def __init__(self, max_log_p=None, min_log_p=None):
        self.max_log_p = float("inf") if max_log_p is None else max_log_p
        self.min_log_p = -float("inf") if min_log_p is None else min_log_p

    #implement
    def mol_check(self, mol: Mol) -> bool:
        log_p = Descriptors.MolLogP(mol)
        return self.min_log_p <= log_p and log_p <= self.max_log_p 