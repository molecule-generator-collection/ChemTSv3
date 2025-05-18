from abc import ABC, abstractmethod
from node import Node

class Policy(ABC):
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def evaluate(self, node: Node, **kwargs):
        pass
    
    def name(self):
        return self.__class__.__name__