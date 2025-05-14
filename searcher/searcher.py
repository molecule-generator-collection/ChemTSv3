from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Type, Any
from reward import Reward, LogPReward

class Searcher(ABC):
  def __init__(self, name=None, reward_class: Type[Reward]=LogPReward, reward_conf: dict=None, output_dir="result", logger_conf: dict[str, Any]=None):
    self._name = name
    self._name = self.name() #generate name if name=None
    self.reward_class = reward_class
    self.reward_conf = reward_conf or {}
    self._output_dir = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep
    os.makedirs(os.path.dirname(self._output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(self.output_dir()), exist_ok=True)
    self.unique_keys = []
    self.record: dict[str, dict] = {} #save at least all of the following for unique molkeys: "objective_values", "reward", "generation_order", "time"
    self.set_logger(logger_conf)

  def name(self):
    if self._name is not None:
      return self._name
    else:
      return self.__class__.__name__ + "_" + datetime.now().strftime("%m-%d_%H-%M")
  
  def output_dir(self):
    return self._output_dir + self.name() + os.sep

  #support more option / yaml later  
  def set_logger(self, logger_conf: dict[str, Any]):
    logger_conf = logger_conf or {}
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)
    
    if not self.logger.handlers:
      console_handler = logging.StreamHandler()
      console_handler.setLevel(logger_conf.get("console_level", logging.INFO))
      file_handler = logging.FileHandler(self.output_dir() + self.name() + ".log")
      file_handler.setLevel(logger_conf.get("file_level", logging.DEBUG))
      self.logger.addHandler(console_handler)
      self.logger.addHandler(file_handler)
      
  #visualize results
  def plot(self, x_axis: str="generation_order", y_axis: str="reward", maxline=False, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None):
    #x_axis ... use X in self.record["mol_key"]["X"]

    x = [self.record[molkey][x_axis] for molkey in self.unique_keys]

    if y_axis == "reward":
      y = [self.record[molkey]["reward"] for molkey in self.unique_keys]
    else:
      objective_names = [f.__name__ for f in self.reward_class.objective_functions()]
      if not y_axis in objective_names:
        self.logger.warning("Couldn't find objective name " + y_axis + ": uses reward instead")
        y_axis = "reward"
        y = [self.record[molkey]["reward"] for molkey in self.unique_keys]
      else:
        objective_id = objective_names.index(y_axis)
        y = [self.record[molkey]["objective_values"][objective_id] for molkey in self.unique_keys]

    plt.clf()
    plt.scatter(x, y, s=1)
    plt.title(self.name())
    
    plt.xlabel(x_axis)
    if xlim is not None:
      plt.xlim(xlim)
    else:
      plt.xlim(0,x[-1])

    plt.ylabel(y_axis)
    if ylim is not None:
      plt.ylim(ylim)
    plt.grid(axis="y")

    if maxline:
      max(y)
      y_max = np.max(y)
      plt.axhline(y=y_max, color='red', linestyle='--', label=f'y={y_max:.5f}')

    plt.savefig(self.output_dir() + self.name() + "_" + y_axis + "_by_" + x_axis + ".png")
    plt.legend()
    plt.show()
  
  def plot_everything(self, x_axis: str="generation_order", maxline=False, xlim: tuple[float, float]=None, ylims: dict[str, tuple[float, float]]=None):
    ylims = ylims or {}
    objective_names = [f.__name__ for f in self.reward_class.objective_functions()]
    for o in objective_names:
      self.plot(x_axis=x_axis, y_axis=o, maxline=maxline, xlim=xlim, ylim=ylims.get(o, None))
    self.plot(x_axis=x_axis, y_axis="reward", maxline=maxline, xlim=xlim, ylim=ylims.get("reward", None))  
