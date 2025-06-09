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
    
    # implement
    def check(self, node: MolNode) -> bool:
        return self.mol_check(node.mol())

class ValueFilter(Filter):
    def __init__(self, max=None, min=None, allowed: int | list[int]=None, disallowed: int | list[int]=None):
        self.max = float("inf") if max is None else max
        self.min = -float("inf") if min is None else min
        if type(allowed) == int:
            allowed = [allowed]
        self.allowed = allowed or []
        if type(disallowed) == int:
            disallowed = [disallowed]
        self.disallowed = disallowed or []

    @abstractmethod
    def value(self, node: Node) -> int | float:
        pass    

    def _check_value(self, value) -> bool:
        for n in self.allowed:
            if value == n:
                return True
            return False
        for n in self.disallowed:
            if value == n:
                return False
        if value < self.min:
            return False
        if value > self.max:
            return False
        return True
    
    # implement
    def check(self, node: Node) -> bool:
        value = self.value(node)
        return self._check_value(value)
    
class MolValueFilter(ValueFilter, MolFilter):
    @abstractmethod
    def mol_value(self, mol: Mol) -> int | float:
        pass
    
    # implement
    def mol_check(self, mol):
        value = self.mol_value(mol)
        return self._check_value(value)
    
    # implement
    def check(self, node: MolNode):
        return self.mol_check(node.mol())
    
    # implement for consistency (not actually needed)
    def value(self, node: MolNode) -> bool:
        return self.mol_value(node.mol())