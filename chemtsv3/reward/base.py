from abc import ABC, abstractmethod
from typing import List, Callable
import numpy as np
from rdkit.Chem import Mol
from chemtsv3.node import Node, MolNode
from chemtsv3.utils import camel2snake

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class Reward(ABC):
    is_single_objective = False

    # abstractmethod
    def objective_functions(self) -> list[Callable[[Node], float | tuple[float] | list[float]]]:
        """
        Return objective functions of the node; each function returns an objective value.
        If any of objective function returns tuple[float], list[float] or 1d ndarray, objective_names() needs to be overridden.
        """
        raise NotImplementedError

    # abstractmethod
    def reward_from_objective_values(self, objective_values: list[float]) -> float:
        """Compute the final reward based on the objective values calculated by objective_functions()."""
        raise NotImplementedError
    
    def objective_values(self, node: Node) -> list[float]:
        values = []
        for f in self.objective_functions():
            v = f(node)
            if isinstance(v, (tuple, list)):
                values.extend(v)
            elif isinstance(v, np.ndarray):
                if v.ndim != 1:
                    raise ValueError(f"{f.__name__} returned ndarray with ndim={v.ndim}, but only 1D arrays are supported.")
                values.extend(v.tolist())
            else:
                values.append(v)
        return values
    
    def objective_names(self) -> list[str]:
        return [f.__name__ for f in self.objective_functions()]
    
    def objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
        objective_values = self.objective_values(node)
        reward = self.reward_from_objective_values(objective_values)
        return objective_values, reward
    
    def objective_values_and_rewards(self, nodes: list[Node]) -> list[tuple[list[float], float]]:
        return [self.objective_values_and_reward(n) for n in nodes]
    
    def n_batch(self) -> int:
        return 1
    
    def is_batch_reward(self) -> int:
        return self.n_batch() > 1
    
    def name(self):
        """(Optional) Override this method to change reward's name displayed on plots."""
        return camel2snake(self.__class__.__name__)
    
    def analyze(self):
        """(Optional) This method is called within Generation.analyze(). By default, this method does nothing."""

class SingleReward(Reward):
    is_single_objective = True
    
    @abstractmethod
    def reward(self, node: Node) -> float:
        pass
    
    # implement
    def objective_functions(self) -> list[Callable[[Node], float]]:
        def r(node):
            return self.reward(node)
        return [r]
    
    # implement
    def reward_from_objective_values(self, objective_values: list[float]) -> float:
        return objective_values[0]

class MolReward(Reward):
    @abstractmethod
    def mol_objective_functions(self) -> list[Callable[[Mol], float]]:
        """Return objective functions of the molecule; each function returns an objective value."""
        pass

    @abstractmethod
    def reward_from_objective_values(self, objective_values: list[float]) -> float:
        """Compute the final reward based on the objective values calculated by objective_functions()."""
        pass

    @staticmethod
    def wrap_with_mol(f):
        def wrapper(node: Node):
            return f(node.mol(save_cache=True))
        wrapper.__name__ = f.__name__ # copy function names
        return wrapper

    #override
    def objective_functions(self) -> list[Callable[[MolNode], float]]:
        return [MolReward.wrap_with_mol(f) for f in self.mol_objective_functions()]
    
class SingleMolReward(SingleReward):
    @abstractmethod
    def mol_reward(self, mol: Mol) -> float:
        pass
    
    # implement
    def reward(self, node: Node) -> float:
        return self.mol_reward(node.mol(save_cache=True))
    
class SMILESReward(Reward):
    @abstractmethod
    def smiles_objective_functions(self) -> list[Callable[[str], float]]:
        pass

    @abstractmethod
    def reward_from_objective_values(self, objective_values: list[float]) -> float:
        pass

    @staticmethod
    def wrap_with_smiles(f):
        def wrapper(node: Node):
            return f(node.smiles(save_cache=True))
        wrapper.__name__ = f.__name__ # copy function names
        return wrapper

    #override
    def objective_functions(self) -> list[Callable[[MolNode], float]]:
        return [SMILESReward.wrap_with_smiles(f) for f in self.smiles_objective_functions()]
    
class BatchReward(Reward, ABC):
    # abstractmethod
    def n_batch(self) -> int:
        """
        Return the number of nodes to simultaneously calculate the reward.
        """
        raise NotImplementedError

    # abstractmethod
    def objective_values_and_rewards(self, nodes: list[Node]) -> list[tuple[list[float], float]]:
        """
        Return [(objective_values, reward), ...] aligned with input nodes.
        """
        raise NotImplementedError
    
    # abstractmethod
    def objective_names(self):
        raise NotImplementedError
    
    def objective_values_and_reward(self, node: Node):
        return self.objective_values_and_rewards([node])[0]