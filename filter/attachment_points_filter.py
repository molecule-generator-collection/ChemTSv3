from chemtsv3.filter import ValueFilter
from chemtsv3.node import Node

class AttachmentPointsFilter(ValueFilter):
    """
    Filter for linker molecules. Node class needs to have smiles() method.
    """
    def __init__(self, allowed=2, **kwargs):
        super().__init__(allowed=allowed, **kwargs)
        
    # implement
    def value(self, node: Node) -> int:
        smiles = node.smiles()
        return smiles.count("*")