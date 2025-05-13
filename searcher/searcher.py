from abc import ABC, abstractmethod

class searcher(ABC):
  @abstractmethod
  def name(self):
    pass