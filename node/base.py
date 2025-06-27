from abc import ABC, abstractmethod
from typing import Self, Any
import numpy as np
from rdkit.Chem import Mol

class Node(ABC):
    use_cache = True
    initial_best_r = -1.0
    
    def __init__(self, parent=None, last_action: Any=None, last_prob=1.0):
        self.parent = parent
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.children: dict[Any, Self] = {}
        self.last_prob = last_prob # Prob(parent -> this node)
        self.last_action = last_action
        self.n = 0 # visit count
        self.sum_r = 0 # sum of rewards
        self.best_r = self.initial_best_r
        self._cache = {} # str, Any
    
    @abstractmethod
    def key(self) -> str:
        pass
    
    def __str__(self) -> str:
        return self.key()

    @abstractmethod
    def is_terminal(self) -> bool:
        pass
    
    @classmethod
    def node_from_key(cls, string: str) -> Self:
        raise NotImplementedError("node_from_key() is not supported in this class.")

    def add_child(self, action: Any, child: Self, override_child=False, override_parent=False):
        if override_child is False and action in self.children:
            pass
        self.children[action] = child
        if child.parent is None or override_parent:
            child.parent = self
    
    def observe(self, value: float):
        self.n += 1
        self.sum_r += value
        self.best_r = max(self.best_r, value)
    
    def sample_children(self, max_size: int=1, replace: bool=False):
        nodes = list(self.children.values())
        size = min(max_size, len(nodes))
        if not nodes:
            return None
        weights = np.array([node.last_prob for node in nodes], dtype=np.float64)
        total = weights.sum()
        probabilities = weights / total
        return np.random.choice(nodes, size=size, replace=replace, p=probabilities)
    
    def sample_child(self):
        return self.sample_children(max_size=1)[0]
    
    def sample_offspring(self, depth: int=1):
        if depth == 1:
            return self.sample_children(max_size=1)[0]
        else:
            return self.sample_children(max_size=1)[0].sample_offspring(depth=depth-1)
        
    def cut_unvisited_children(self):
        unvisited_keys = [key for key, child in self.children.items() if child.n == 0]
        for key in unvisited_keys:
            del self.children[key]
        
    def show_children(self):
        for child in sorted(self.children.values(), key=lambda c: c.last_prob, reverse=True):
            print(f"{child.last_prob:.3f}", str(child))
    
    def clear_cache(self):
        self._cache = {}
    
class MolNode(Node):
    @abstractmethod
    def key(self) -> str:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def _mol_impl(self) -> Mol:
        pass
    
    def mol(self) -> Mol:
        if self.use_cache and "mol" in self._cache:
            return self._cache["mol"]
        else:
            mol = self._mol_impl()
            self._cache["mol"] = mol
            return mol
        
class SurrogateNode(Node):
    """surrogate node for multiple roots"""
    # implement
    def key(self) -> str:
        return "surrogate node"

    # implement
    def is_terminal(self) -> bool:
        return False