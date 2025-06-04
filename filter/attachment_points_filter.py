from rdkit.Chem import Mol, Descriptors
from filter import Filter
from node import SentenceNode

class AttachmentPointsFilter(Filter):
    def __init__(self, n=2):
        self.n = n
        
    #implement
    def check(self, node: SentenceNode) -> bool:
        smiles = str(node)
        return smiles.count("*") == self.n