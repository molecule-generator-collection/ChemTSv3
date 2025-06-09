from abc import ABC, abstractmethod
from typing import Self, Any
from rdkit.Chem import Mol

class Node(ABC):
    def __init__(self, parent=None, last_action: Any=None, last_prob=1.0, use_cache=True):
        self.parent = parent
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.children: dict[Any, Self] = {}
        self._probs = [] # call probs()
        self.last_prob = last_prob # Prob(parent -> this node)
        self.last_action = last_action
        self.n = 0 # visit count
        self.sum_r = 0.0 # sum of rewards
        self.mean_r = 0.0 # mean of rewards
        self.use_cache = use_cache
        self._cache = {} # str, Any

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    def add_child(self, action: Any, child: Self):
        self.children[action] = child
    
    def observe(self, value: float):
        self.n += 1
        self.sum_r += value
        self.mean_r = self.sum_r / self.n
    
    def clear_cache(self):
        self._cache = {}
    
class MolNode(Node):
    @abstractmethod
    def __str__(self) -> str:
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