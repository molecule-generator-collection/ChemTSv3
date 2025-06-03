from rdkit.Chem import Mol, Descriptors
from filter import MolFilter

class RingSizeFilter(MolFilter):
    def __init__(self, max_ring_size = None, min_ring_size = None):
        self.max_ring_size = max_ring_size
        self.min_ring_size = min_ring_size

    #implement
    def mol_check(self, mol: Mol) -> bool:
        ri = mol.GetRingInfo()
        if self.max_ring_size is not None and self.max_ring_size < max((len(r) for r in ri.AtomRings()), default=0):
            return False
        if self.min_ring_size is not None and self.min_ring_size > min((len(r) for r in ri.AtomRings()), default=float("inf")):
            return False
        return True