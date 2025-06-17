from abc import ABC, abstractmethod
from typing import Self, Any
import numpy as np
from rdkit.Chem import Mol

class Node(ABC):
    use_cache = True
    
    def __init__(self, parent=None, last_action: Any=None, last_prob=1.0):
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
        self._cache = {} # str, Any

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    def add_child(self, action: Any, child: Self, override=False):
        if override is False and action in self.children:
            pass
        self.children[action] = child
    
    def observe(self, value: float):
        self.n += 1
        self.sum_r += value
        
    def mean_r(self):
        return self.sum_r / self.n
        
    def sample_child(self, additional_depth: int=1) -> Self:
        if additional_depth == 1:
            nodes = list(self.children.values())
            if not nodes:
                return None
            weights = np.array([node.last_prob for node in nodes], dtype=np.float64)
            total = weights.sum()
            probabilities = weights / total
            return np.random.choice(nodes, p=probabilities)
        else:
            return self.sample_child().sample_child(additional_depth=additional_depth-1)
        
    def show_children(self):
        for child in sorted(self.children.values(), key=lambda c: c.last_prob, reverse=True):
            print(f"{child.last_prob:.3f}", str(child))
    
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