from abc import ABC, abstractmethod
from rdkit.Chem import Mol
from node import Node, MolNode

class Filter(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def check(self, node: Node) -> bool:
        pass

class MolFilter(Filter):
    @abstractmethod
    def mol_check(self, mol: Mol) -> bool:
        pass
    
    #implement
    def check(self, node: MolNode) -> bool:
        return self.mol_check(node.mol())