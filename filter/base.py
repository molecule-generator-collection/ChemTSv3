from abc import ABC, abstractmethod
from rdkit.Chem import Mol
from node import Node, MolNode

class Filter(ABC):
    @abstractmethod
    def check(self, node: Node) -> bool | float:
        """Return False to skip reward calculation for the given node. Return float value to override the reward. When applied for transition, float value won't be used, and will be considered False."""
        pass
    
    def observe(self, node: Node, objective_values: list[float], reward: float, is_filtered: bool):
        """Filters can update their internal state when observing the reward of the node. By default, this method does nothing."""
        return
    
    def on_inherit(self, generator):
        """This method is called after inheriting the generator states on chain generation. By default, this method does nothing."""

    def analyze(self):
        """This method is called within Generation.analyze(). By default, this method does nothing."""

class MolFilter(Filter):
    """Filter for MolNode"""
    @abstractmethod
    def mol_check(self, mol: Mol) -> bool | float:
        """Return False to skip reward calculation for the given molecule."""
        pass
    
    # implement
    def check(self, node: MolNode) -> bool | float:
        return self.mol_check(node.mol(use_cache=True))

class ValueFilter(Filter):
    """Filter that excludes nodes based on a single numerical value."""
    def __init__(self, max: float=None, min: float=None, allowed: int | list[int]=None, disallowed: int | list[int]=None):
        """
        Args:
            max: Nodes with values higher than this will be filtered.
            min: Nodes with values lower than this will be filtered.
            allowed: List of explicitly allowed values. If provided, nodes with values not in this list will be filtered.
            disallowed: List of explicitly disallowed values. If provided, nodes with values in this list will be filtered.
        """
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
    """Filter that excludes nodes based on a single numerical value of the molecule."""
    @abstractmethod
    def mol_value(self, mol: Mol) -> int | float:
        pass
    
    # implement
    def mol_check(self, mol):
        value = self.mol_value(mol)
        return self._check_value(value)
    
    # implement
    def check(self, node: MolNode):
        return self.mol_check(node.mol(use_cache=True))
    
    # implement for consistency (not actually needed)
    def value(self, node: MolNode) -> bool:
        return self.mol_value(node.mol(use_cache=True))