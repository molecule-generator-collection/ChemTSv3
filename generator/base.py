from abc import ABC, abstractmethod
from datetime import datetime
import logging
import math
import os
import time
from typing import Type, Any
import matplotlib.pyplot as plt
import numpy as np
from filter import Filter
from node import Node
from reward import Reward, LogPReward
from utils import camel2snake

class Generator(ABC):
    def __init__(self, output_dir="generation_result", name=None, reward: Reward=LogPReward(), filters: list[Filter]=None, filtered_reward: float=0, logger_conf: dict[str, Any]=None):
        # transition is not passed: generator with multiple transition rules
        self._name = name
        self._name = self.name() # generate name if name=None
        self.reward: Reward = reward
        self.filters: list[Filter] = filters or []
        self.filtered_reward = filtered_reward
        self._output_dir = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep
        os.makedirs(os.path.dirname(self._output_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_dir()), exist_ok=True)
        self.unique_keys = []
        self.record: dict[str, dict] = {} # save at least all of the following for unique molkeys: "objective_values", "reward", "generation_order", "time"
        self.passed_time = 0
        self.set_logger(logger_conf)
    
    @abstractmethod
    def _generate_impl(self, *kwargs):
        pass

    def generate(self, time_limit: float=None, max_generations: int=None):
        """
        Generate nodes that either is_terminal() = True or depth = max_length. Tries to maximize the reward by MCTS search.

        Args:
            time_limit: Seconds. Generation stops after the time limit.
            max_generations: Generation stops after generating 'max_generations' number of nodes.
        """
        if (time_limit is None) and (max_generations is None):
            raise ValueError("Specify at least one of max_genrations, max_rollouts or time_limit.")
        
        # record current time and counts
        time_start = time.time()
        initial_time = self.passed_time
        initial_count_generations = len(self.unique_keys)
        
        self.logger.info("Search is started.")
        while True:
            time_passed = time.time() - time_start
            self.passed_time = initial_time + time_passed
            if time_limit is not None and time_passed >= time_limit:
                break
            if max_generations is not None and len(self.unique_keys) - initial_count_generations >= max_generations:
                break
            
            self._generate_impl()
            
        self.logger.info("Search is completed.")

    def name(self):
        if self._name is not None:
            return self._name
        else:
            return datetime.now().strftime("%m-%d_%H-%M") + "_" + self.__class__.__name__
    
    def output_dir(self):
        return self._output_dir + self.name() + os.sep

    # support more option / yaml later
    def set_logger(self, logger_conf: dict[str, Any]):
        logger_conf = logger_conf or {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger_conf.get("console_level", logging.INFO))
        file_handler = logging.FileHandler(self.output_dir() + self.name() + ".log")
        file_handler.setLevel(logger_conf.get("file_level", logging.DEBUG))
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_unique_node(self, key, objective_values, reward):
        self.logger.info(str(len(self.unique_keys)) + "- time: " + "{:.2f}".format(self.passed_time) + ", reward: " + str(reward) + ", node: " + key)
        self.unique_keys.append(key)
        self.record[key] = {}
        self.record[key]["objective_values"] = objective_values
        self.record[key]["reward"] = reward
        self.record[key]["time"] = self.passed_time
        self.record[key]["generation_order"] = len(self.unique_keys)

    def grab_objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
        key = str(node)
        if key in self.record:
            self.logger.debug("Already in dict: " + key + ", count_rollouts: " + str(self.count_rollouts) + ", reward: " + str(self.record[key]["reward"]))
            return self.record[key]["objective_values"], self.record[key]["reward"]
        
        for filter in self.filters:
            if not filter.check(node):
                self.logger.debug("filtered by " + filter.__class__.__name__ + ": " + key)
                return ([0,0], self.filtered_reward)
            
        objective_values, reward = self.reward.objective_values_and_reward(node)
        self.log_unique_node(key, objective_values, reward)
        node.clear_cache()

        return objective_values, reward

    # visualize results
    def plot_objective_values_and_reward(self, x_axis: str="generation_order", moving_average: int | float=0.05, max_curve=True, max_line=False, xlim: tuple[float, float]=None, ylims: dict[str, tuple[float, float]]=None):
        ylims = ylims or {}
        objective_names = [f.__name__ for f in self.reward.objective_functions()]
        for o in objective_names:
            self._plot(x_axis=x_axis, y_axis=o, max_line=max_line, xlim=xlim, ylim=ylims.get(o, None))
        self._plot(x_axis=x_axis, y_axis="reward", moving_average=moving_average, max_curve=max_curve, max_line=max_line, xlim=xlim, ylim=ylims.get("reward", None))

    def _plot(self, x_axis: str="generation_order", y_axis: str="reward", moving_average: int | float=0.05, max_curve=True, max_line=False, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None):
        # x_axis ... use X in self.record["mol_key"]["X"]

        x = [self.record[molkey][x_axis] for molkey in self.unique_keys]

        if y_axis == "reward":
            y_axis = camel2snake(self.reward.__class__.__name__)
            y = [self.record[molkey]["reward"] for molkey in self.unique_keys]
        else:
            objective_names = [f.__name__ for f in self.reward.objective_functions()]
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
        
        label = f"moving average ({moving_average})"
        if moving_average is not None and moving_average < 1:
            moving_average = math.floor(len(self.unique_keys) * moving_average)
        if moving_average is not None and moving_average > 1:
            y_ma = np.convolve(y, np.ones(moving_average) / moving_average, mode='valid')
            x_ma = x[moving_average - 1:]  # align with shorter y_ma
            plt.plot(x_ma, y_ma, label=label, linewidth=1.5)

        if max_curve:
            y_max_curve = np.maximum.accumulate(y)
            plt.plot(x, y_max_curve, label='max', linestyle='--')

        if max_line:
            max(y)
            y_max = np.max(y)
            plt.axhline(y=y_max, color='red', linestyle='--', label=f'y={y_max:.5f}')

        plt.legend()
        plt.savefig(self.output_dir() + self.name() + "_" + y_axis + "_by_" + x_axis + ".png")
        plt.show()
