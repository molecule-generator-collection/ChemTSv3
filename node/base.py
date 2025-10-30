from abc import ABC, abstractmethod
import weakref
import logging
from typing import Self, Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, inchi

class Node(ABC):
    initial_best_r = 0.0
    
    def __init__(self, parent=None, last_action: Any=None, last_prob=1.0):
        # self.parent = parent
        self._parent_ref = weakref.ref(parent) if parent is not None else None
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.children = []
        self.last_prob = last_prob # Prob(parent -> this node)
        self.last_action = last_action
        self.n = 0 # visit count
        self.sum_r = 0.0 # sum of rewards
        self.best_r = self.initial_best_r
        self.reward = None # used only if has_reward() = True
        self._is_terminal = False # set this to True in generator if transition_with_probs returned an empty list
        self._cache = None # use self.cache and self.clear_cache() (dict)
    
    @abstractmethod
    def key(self) -> str:
        """Return the key string. Keys are used for identity checks between nodes, and the keys of generated nodes will be recorded."""
        pass
    
    @abstractmethod
    def has_reward(self) -> bool:
        """Define the reward condition."""
        pass
    
    # should be overridden for YAML compatibility etc.
    @classmethod
    def node_from_key(cls, key: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        """
        Create a Node instance from a key.
        For a minimal YAML compatibility, a starting node for empty string "" should be defined here. (ex: if key=="" return ~)
        For YAML root specification / auto chain generation, nodes correspond to possible keys (string values) should be defined here.
        """
        raise NotImplementedError("node_from_key() is not supported in this class.")
    
    # def hash(self):
    #     return self.key()
    
    def discard_unneeded_states(self):
        """Clear states no longer needed after transition to reduce memory usage. Can be overridden for marginal efficiency."""

        needed = ["parent", "depth", "children", "last_prob", "last_action", "n", "sum_r", "best_r", "reward", "_is_terminal"]
        for key in list(self.__dict__.keys()):
            if key not in needed:
                self.__dict__[key] = None
    
    @property
    def cache(self):
        if self._cache is None:
            self._cache = {}
        return self._cache
    
    @property
    def parent(self):
        return self._parent_ref() if self._parent_ref is not None else None
    
    @parent.setter
    def parent(self, p):
        self._parent_ref = weakref.ref(p) if p is not None else None
    
    def clear_cache(self):
        self._cache = None
    
    def mark_as_terminal(self, cut=False, logger: logging.Logger=None) -> bool:
        self._is_terminal = True
        if cut:
            self.leave(recursive=True, logger=logger)

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
    
    def sample_child(self) -> Self:
        return self.sample_children(max_size=1)[0]
    
    def sample_offspring(self, depth: int=1) -> Self:
        if depth == 1:
            return self.sample_children(max_size=1)[0]
        else:
            return self.sample_children(max_size=1)[0].sample_offspring(depth=depth-1)
        
    def count_offsprings(self) -> int:
        count = 0
        stack = [self]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count
        
    def leave(self, recursive=True, logger: logging.Logger=None):
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)
            if recursive and not self.parent.children:
                # if logger is not None:
                #     logger.debug(f"Exhausted every terminal under: {self.parent.key()}") # Set discard_unneeded_children to False when commenting out
                self.parent.leave(recursive=True, logger=logger)
        
    def show_children(self):
        for child in sorted(self.children, key=lambda c: c.last_prob, reverse=True):
            print(f"{child.last_prob:.3f}, {str(child.key())}")
        
    def __str__(self) -> str:
        return self.key()
    
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_parent_ref"] = self.parent
        return s

    def __setstate__(self, state):
        parent_obj = state.get("_parent_ref", None)
        state["_parent_ref"] = weakref.ref(parent_obj) if parent_obj is not None else None
        self.__dict__.update(state)

class MolNode(Node):
    # use_inchikey_as_hash = False # Not fully implemented / tested yet
    
    @abstractmethod
    def key(self) -> str:
        pass
    
    @abstractmethod
    def has_reward(self) -> bool:
        pass

    @abstractmethod
    def _mol_impl(self) -> Mol:
        pass
    
    def mol(self, use_cache=False) -> Mol:
        if "mol" in self.cache:
            return self.cache["mol"]
        if not use_cache:
            return self._mol_impl()
        else:
            mol = self._mol_impl()
            self.cache["mol"] = mol
            return mol
        
    def smiles(self, use_cache=False) -> str:
        """Should be overridden if the node has an explicit SMILES as a variable."""
        return Chem.MolToSmiles(self.mol(use_cache=use_cache))
    
    # override
    # def hash(self):
    #     if not self.use_inchikey_as_hash:
    #         return super().hash()
    #     else:
    #         return inchi.MolToInchiKey(self.mol()) 

class SurrogateNode(Node):
    """Surrogate node for multiple roots."""
    def key(self) -> str:
        return "surrogate node"
    
    def has_reward(self) -> bool:
        return False