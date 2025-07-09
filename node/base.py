from abc import ABC, abstractmethod
from typing import Self, Any
import numpy as np
from rdkit.Chem import Mol

class Node(ABC):
    initial_best_r = -1.0
    
    def __init__(self, parent=None, last_action: Any=None, last_prob=1.0):
        self.parent = parent
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.children = []
        self.last_prob = last_prob # Prob(parent -> this node)
        self.last_action = last_action
        self.n = 0 # visit count
        self.sum_r = 0 # sum of rewards
        self.best_r = self.initial_best_r
        self.reward = None # used only if has_reward() = True
        self._is_terminal = False # set this to True in generator if transition_with_probs returned an empty list
        self._cache = None # use self.cache and self.clear_cache() (dict)
    
    @abstractmethod
    def key(self) -> str:
        pass
    
    @abstractmethod
    def has_reward(self) -> bool:
        pass
    
    # should be overridden if root specification is needed
    @classmethod
    def node_from_key(cls, key: str) -> Self:
        raise NotImplementedError("node_from_key() is not supported in this class.")
    
    @property
    def cache(self):
        if self._cache is None:
            self._cache = {}
        return self._cache
    
    def clear_cache(self):
        self._cache = None
    
    def mark_as_terminal(self, freeze=False) -> bool:
        self._is_terminal = True
        if freeze:
            self.n += 1 # to avoid n=0 score
            self.sum_r = -float("inf")

    def is_terminal(self) -> bool:
        return self._is_terminal

    def add_child(self, child: Self, override_parent=False):
        self.children.append(child)
        if child.parent is None or override_parent:
            child.parent = self
    
    def observe(self, value: float):
        self.n += 1
        self.sum_r += value
        self.best_r = max(self.best_r, value)
    
    def sample_children(self, max_size: int=1, replace: bool=False):
        if not self.children:
            return None
        size = min(max_size, len(self.children))
        weights = np.array([node.last_prob for node in self.children], dtype=np.float64)
        total = weights.sum()
        probabilities = weights / total
        return np.random.choice(self.children, size=size, replace=replace, p=probabilities)
    
    def sample_child(self):
        return self.sample_children(max_size=1)[0]
    
    def sample_offspring(self, depth: int=1):
        if depth == 1:
            return self.sample_children(max_size=1)[0]
        else:
            return self.sample_children(max_size=1)[0].sample_offspring(depth=depth-1)

    def cut_unvisited_children(self):
        self.children = [node for node in self.children if node.n != 0]
        
    def leave(self):
        if self in self.parent.children:
            self.parent.children.remove(self)
        
    def show_children(self):
        for child in sorted(self.children, key=lambda c: c.last_prob, reverse=True):
            print(f"{child.last_prob:.3f}", str(child))
        
    def __str__(self) -> str:
        return self.key()
    
class MolNode(Node):
    @abstractmethod
    def key(self) -> str:
        pass
    
    @abstractmethod
    def has_reward(self) -> bool:
        pass

    @abstractmethod
    def _mol_impl(self) -> Mol:
        pass
    
    def mol(self, use_cache=True) -> Mol:
        if not use_cache:
            return self._mol_impl()
        if "mol" in self.cache:
            return self.cache["mol"]
        else:
            mol = self._mol_impl()
            self.cache["mol"] = mol
            return mol

class SurrogateNode(Node):
    """Surrogate node for multiple roots."""
    # implement
    def key(self) -> str:
        return "surrogate node"
    
    # implement
    def has_reward(self) -> bool:
        return False