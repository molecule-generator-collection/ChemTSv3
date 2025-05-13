from abc import ABC, abstractmethod
import time, datetime

class Searcher(ABC):
  def __init__(self, name=None):
    self._name = name
    self._name = self.name() #generate name if name=None

  def name(self):
    if self._name is not None:
      return self._name
    else:
      return str(datetime.datetime.now())
  
