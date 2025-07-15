from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class MaxRingSizeFilter(MolValueFilter):
    def __init__(self, max=6, min=None, allowed=None, disallowed=None):
        super().__init__(allowed=allowed, max=max, min=min, allowed=allowed, disallowed=disallowed)
        
    # implement
    def mol_value(self, mol: Mol) -> int:
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        return max_ring_size
    
class MinRingSizeFilter(MolValueFilter):
    # implement
    def mol_value(self, mol: Mol) -> int:
        ri = mol.GetRingInfo()
        min_ring_size = min((len(r) for r in ri.AtomRings()), default=float("inf"))
        return min_ring_size