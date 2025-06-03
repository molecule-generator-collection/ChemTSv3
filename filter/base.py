from abc import ABC, abstractmethod
from node import Node, MolNode

class Filter(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def check(self, node: Node) -> bool:
        pass

class MolFilter(Filter):
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def check(self, node: MolNode) -> bool:
        pass