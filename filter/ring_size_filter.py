from rdkit.Chem import Mol, Descriptors
from filter import MolValueFilter

class MaxRingSizeFilter(MolValueFilter):
    #implement
    def mol_value(self, mol: Mol) -> bool:
        ri = mol.GetRingInfo()
        max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        return max_ring_size
    
class MinRingSizeFilter(MolValueFilter):
    #implement
    def mol_value(self, mol: Mol) -> bool:
        ri = mol.GetRingInfo()
        min_ring_size = min((len(r) for r in ri.AtomRings()), default=float("inf"))
        return min_ring_size