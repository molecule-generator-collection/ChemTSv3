from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

class Searcher(ABC):
  def __init__(self, name=None, print_output=True, output_dir="result"):
    self._name = name
    self._name = self.name() #generate name if name=None
    self.print_output = print_output
    self._output_dir = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep
    os.makedirs(os.path.dirname(self._output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(self.output_dir()), exist_ok=True)
    self.unique_molkeys = []
    self.record: dict[str, dict] = {} #save at least all of the following for unique molkeys: "objective_values", "reward", "generation_order", "time"

  def name(self):
    if self._name is not None:
      return self._name
    else:
      return datetime.now().strftime("%m-%d_%H-%M")
  
  def output_dir(self):
    return self._output_dir + self.name() + os.sep
  
  #print_output
  def logging(self, str, force=False):
    if self.print_output or force:
      print(str)
    with open(self.output_dir() + self.name() + ".txt", "a") as f:
      f.write(str + "\n")
      
  #visualize results
  def plot(self, x_axis: str = "generation_order", maxline = False, xlim: tuple[float, float] = None, ylim: tuple[float, float] = None):
    #x_axis ... use X in self.record["mol_key"]["X"]

    x = [self.record[molkey][x_axis] for molkey in self.unique_molkeys]
    y = [self.record[molkey]["reward"] for molkey in self.unique_molkeys]

    plt.clf()
    plt.scatter(x, y, s=1)
    plt.title(self.name())
    
    if xlim is not None:
      plt.xlim(xlim)
    else:
      plt.xlim(0,x[-1])
    plt.xlabel(x_axis)

    if ylim is not None:
      plt.ylim(ylim)
    plt.ylabel("reward")
    plt.grid(axis="y")

    if maxline:
      max(y)
      y_max = np.max(y)
      plt.axhline(y=y_max, color='red', linestyle='--', label=f'y={y_max:.5f}')

    plt.savefig(self.output_dir() + self.name() + ".png")
    plt.legend()
    plt.show()
  
