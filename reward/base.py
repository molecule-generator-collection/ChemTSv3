from abc import ABC, abstractmethod
from typing import List, Callable
from rdkit.Chem import Mol
from node import Node, MolNode
from utils import camel2snake

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class Reward(ABC):
    is_single_objective = False

    @abstractmethod
    def objective_functions(self) -> List[Callable[[Node], float]]:
        """Return objective functions of the node; each function returns an objective value."""
        pass

    @abstractmethod
    def reward_from_objective_values(self, objective_values: List[float]) -> float:
        """Compute the final reward based on the objective values calculated by objective_functions()."""
        pass

    def objective_values(self, node: Node):
        return [f(node) for f in self.objective_functions()]
    
    def objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
        objective_values = self.objective_values(node)
        reward = self.reward_from_objective_values(objective_values)
        return objective_values, reward
    
    def name(self):
        return camel2snake(self.__class__.__name__)

class MolReward(Reward):
    @abstractmethod
    def mol_objective_functions(self) -> List[Callable[[Mol], float]]:
        """Return objective functions of the molecule; each function returns an objective value."""
        pass

    @abstractmethod
    def reward_from_objective_values(self, objective_values: List[float]) -> float:
        """Compute the final reward based on the objective values calculated by objective_functions()."""
        pass

    @staticmethod
    def wrap_with_mol(f):
        def wrapper(node: Node):
            return f(node.mol(use_cache=True))
        wrapper.__name__ = f.__name__ # copy function names
        return wrapper

    #override
    def objective_functions(self) -> List[Callable[[MolNode], float]]:
        return [MolReward.wrap_with_mol(f) for f in self.mol_objective_functions()]
    
class SMILESReward(Reward):
    @abstractmethod
    def smiles_objective_functions(self) -> List[Callable[[str], float]]:
        pass

    @abstractmethod
    def reward_from_objective_values(self, objective_values: List[float]) -> float:
        pass

    @staticmethod
    def wrap_with_smiles(f):
        def wrapper(node: Node):
            return f(node.smiles(use_cache=True))
        wrapper.__name__ = f.__name__ # copy function names
        return wrapper

    #override
    def objective_functions(self) -> List[Callable[[MolNode], float]]:
        return [SMILESReward.wrap_with_smiles(f) for f in self.smiles_objective_functions()]