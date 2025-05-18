from abc import ABC, abstractmethod
from typing import Any, List, Callable
from rdkit.Chem import Mol
from node import Node, MolNode

class Reward(ABC):
    #Callable[[Node], float] instead of Callable[[Mol], float] for better compatibility
    def __init__(**kwargs):
        pass
    
    @abstractmethod
    def objective_functions(self) -> List[Callable[[Node], float]]:
        pass

    @abstractmethod
    def reward_from_objective_values(self, values: List[float]) -> float:
        pass

    def objective_values(self, node: Node):
        return [f(node) for f in self.objective_functions()]
    
    def objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
        objective_values = self.objective_values(node)
        reward = self.reward_from_objective_values(objective_values)
        return objective_values, reward

class MolReward(Reward):
    @abstractmethod
    def mol_objective_functions(self) -> List[Callable[[Mol], float]]:
        pass

    @abstractmethod
    def reward_from_objective_values(self, values: List[float]) -> float:
        pass

    @staticmethod
    def wrap_with_mol(f):
        def wrapper(node: Node):
            return f(node.mol())
        wrapper.__name__ = f.__name__ #copy function names
        return wrapper

    #override
    def objective_functions(self) -> List[Callable[[MolNode], float]]:
        return [MolReward.wrap_with_mol(f) for f in self.mol_objective_functions()]