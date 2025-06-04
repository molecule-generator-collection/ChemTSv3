from rdkit.Chem import Mol, Descriptors
from filter import ValueFilter
from node import SentenceNode

class AttachmentPointsFilter(ValueFilter):
    def __init__(self, allowed=2, **kwargs):
        super().__init__(allowed=allowed, **kwargs)
        
    #implement
    def check(self, node: SentenceNode) -> bool:
        smiles = str(node)
        return smiles.count("*")