from abc import ABC, abstractmethod
from datetime import datetime
import logging
import os
import pickle
import time
from typing import Type, Any, Self
import matplotlib.pyplot as plt
import numpy as np
from filter import Filter
from node import Node
from reward import Reward, LogPReward
from transition import Transition
from utils import camel2snake, moving_average, make_logger

class Generator(ABC):
    def __init__(self, transition: Transition, output_dir="generation_result", name=None, reward: Reward=LogPReward(), filters: list[Filter]=None, filtered_reward: float=0, logger: logging.Logger=None, info_interval: int=1):
        # when implementing generator with multiple transition rules, add list[Transition] to type hint
        self.transition = transition
        self._name = name or self.make_name()
        self.reward: Reward = reward
        self.filters: list[Filter] = filters or []
        self.filtered_reward = filtered_reward
        self._output_dir = output_dir
        os.makedirs(self.output_dir(), exist_ok=True)
        self.unique_keys = []
        self.record: dict[str, dict] = {} # save at least all of the following for unique molkeys: "objective_values", "reward", "generation_order", "time"
        self.best_reward = -float("inf")
        self.passed_time = 0
        self.grab_count = 0
        self.duplicate_count = 0
        self.filtered_count = 0
        self.logger = logger or make_logger(output_dir=self.output_dir(), name=self.name())
        self.info_interval = info_interval
    
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
        if self.passed_time == 0:
            self.write_csv_header()
        initial_count_generations = len(self.unique_keys)
        
        self.logger.info("Starting generation...")
        try:
            while True:
                time_passed = time.time() - time_start
                self.passed_time = initial_time + time_passed
                if time_limit is not None and time_passed >= time_limit:
                    break
                if max_generations is not None and len(self.unique_keys) - initial_count_generations >= max_generations:
                    break
                self._generate_impl()
        except KeyboardInterrupt:
            self.logger.warning("Generation interrupted by user (KeyboardInterrupt).")
        finally:
            if hasattr(self, "executor"): # for MP
                self.executor.shutdown(cancel_futures=True)
                self.logger.info("Executor shutdown completed.")
            self.logger.info("Generation finished.")

    def make_name(self):
        return datetime.now().strftime("%m-%d_%H-%M") + "_" + self.__class__.__name__
    
    def name(self):
        return self._name
    
    def output_dir(self):
        return self._output_dir if self._output_dir.endswith(os.sep) else self._output_dir + os.sep

    def write_csv_header(self):
        header = ["order", "time", "key"]
        header.append(camel2snake(self.reward.__class__.__name__))
        header += [f.__name__ for f in self.reward.objective_functions()]
        self.logger.info(header)

    def log_unique_node(self, key, objective_values, reward):
        self.unique_keys.append(key)
        self.record[key] = {}
        self.record[key]["objective_values"] = objective_values
        self.record[key]["reward"] = reward
        self.record[key]["time"] = self.passed_time
        self.record[key]["generation_order"] = len(self.unique_keys)
        
        if self.info_interval <= 1 or reward > self.best_reward:
            if reward > self.best_reward:
                self.best_reward = reward
                prefix = "<best reward updated> "
            else:
                prefix = ""
            self.logger.info(prefix + "order: " + str(len(self.unique_keys)) + ", time: " + "{:.2f}".format(self.passed_time) + ", reward: " + "{:.4f}".format(reward) + ", node: " + key)
        else:
            if len(self.unique_keys)%self.info_interval == 0:
                rewards = [self.record[k]["reward"] for k in self.unique_keys[-self.info_interval:]]
                average = np.average(rewards)
                self.logger.info("generated: " + str(len(self.unique_keys)) + ", time: " + "{:.2f}".format(self.passed_time) + ", average over " + str(self.info_interval) + ": " + "{:.4f}".format(average))

        row = [len(self.unique_keys), self.passed_time, key, reward, *objective_values]
        self.logger.info(row)

    def grab_objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
        self.grab_count += 1
        key = str(node)
        if key in self.record:
            self.duplicate_count += 1
            self.logger.debug("already in dict: " + key + ", reward: " + str(self.record[key]["reward"]))
            return self.record[key]["objective_values"], self.record[key]["reward"]
        
        for i, filter in enumerate(self.filters):
            if not filter.check(node):
                self.filtered_count += 1
                self.logger.debug("filtered by " + filter.__class__.__name__ + ": " + key)
                return [str(i)], self.filtered_reward
            
        objective_values, reward = self.reward.objective_values_and_reward(node)
        self.log_unique_node(key, objective_values, reward)
        node.clear_cache()

        return objective_values, reward

    # visualize results
    def plot(self, x_axis: str="generation_order", moving_average_window: int | float=0.01, max_curve=True, max_line=False, xlim: tuple[float, float]=None, ylims: dict[str, tuple[float, float]]=None, packed_objectives=None):
        self._plot_objective_values_and_reward(x_axis=x_axis, moving_average_window=moving_average_window, max_curve=max_curve, max_line=max_line, xlim=xlim, ylims=ylims)
        if packed_objectives:
            for po in packed_objectives:
                self._plot_specified_objective_values(po, x_axis=x_axis, moving_average_window=moving_average_window, xlim=xlim)

    def _plot(self, x_axis: str="generation_order", y_axis: str | list[str]="reward", moving_average_window: int | float=0.01, max_curve=True, max_line=False, scatter=True, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, loc: str="lower right"):
        # x_axis ... use X in self.record["mol_key"]["X"]

        x = [self.record[molkey][x_axis] for molkey in self.unique_keys]

        reward_name = camel2snake(self.reward.__class__.__name__)
        if y_axis == "reward" or y_axis == reward_name:
            y_axis = reward_name
            y = [self.record[molkey]["reward"] for molkey in self.unique_keys]
        else:
            objective_names = [f.__name__ for f in self.reward.objective_functions()]
            if not y_axis in objective_names:
                self.logger.warning("Couldn't find objective name " + y_axis + ": uses reward instead.")
                y_axis = "reward"
                y = [self.record[molkey]["reward"] for molkey in self.unique_keys]
            else:
                objective_idx = objective_names.index(y_axis)
                y = [self.record[molkey]["objective_values"][objective_idx] for molkey in self.unique_keys]

        plt.clf()
        plt.scatter(x, y, s=500/len(x), alpha=0.2)
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
        
        if moving_average_window is not None:
            label = f"moving average ({moving_average_window})"
            y_ma = moving_average(y, moving_average_window)
            plt.plot(x, y_ma, label=label, linewidth=1.5)

        if max_curve:
            y_max_curve = np.maximum.accumulate(y)
            plt.plot(x, y_max_curve, label='max', linestyle='--')

        if max_line:
            max(y)
            y_max = np.max(y)
            plt.axhline(y=y_max, color='red', linestyle='--', label=f'y={y_max:.5f}')
        
        plt.legend(loc=loc)
        plt.savefig(self.output_dir() + self.name() + "_" + y_axis + "_by_" + x_axis + ".png")
        plt.show()
        
    def _plot_objective_values_and_reward(self, x_axis: str="generation_order", moving_average_window: int | float=0.01, max_curve=True, max_line=False, xlim: tuple[float, float]=None, ylims: dict[str, tuple[float, float]]=None, loc: str="lower right"):
        ylims = ylims or {}
        objective_names = [f.__name__ for f in self.reward.objective_functions()]
        for o in objective_names:
            self._plot(x_axis=x_axis, y_axis=o, max_line=max_line, xlim=xlim, ylim=ylims.get(o, None))
        self._plot(x_axis=x_axis, y_axis="reward", moving_average_window=moving_average_window, max_curve=max_curve, max_line=max_line, xlim=xlim, ylim=ylims.get("reward", None), loc=loc)

    def _plot_specified_objective_values(self, y_axes: list[str], x_axis: str="generation_order", moving_average_window: int | float=0.01, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, loc: str="lower right"):
        x = [self.record[molkey][x_axis] for molkey in self.unique_keys]
        objective_names = [f.__name__ for f in self.reward.objective_functions()]
        for ya in y_axes:
            label = ya
            objective_idx = objective_names.index(ya)
            y = [self.record[molkey]["objective_values"][objective_idx] for molkey in self.unique_keys]
            y_ma = moving_average(y, moving_average_window)
            plt.plot(x, y_ma, label=label, linewidth=1.5)
        plt.grid(axis="y")
        plt.title(self.name() + "_ma_window=" + str(moving_average_window))
        plt.legend(loc=loc)
        plt.savefig(self.output_dir() + self.name() + "_by_" + x_axis + ".png")
        plt.show()

    def analyze(self):
        self.logger.info("number of generated nodes: " + str(len(self.unique_keys)))
        valid_rate = 1 - (self.filtered_count / self.grab_count)
        self.logger.info("valid rate: " + str(valid_rate))
        unique_rate = 1 - (self.duplicate_count / self.grab_count)
        self.logger.info("unique rate: " + str(unique_rate))
        node_per_sec = len(self.unique_keys) / self.passed_time
        self.logger.info("node_per_sec: " + str(node_per_sec))
        
    def __getstate__(self):
        state = self.__dict__.copy()
        if "transition" in state:
            del state["transition"]
        return state
    
    def save(self, file: str):
        with open(file, mode="wb") as fo:
            pickle.dump(self, fo)

    def load(file: str, transition: Transition) -> Self:
        with open(file, "rb") as f:
            generator = pickle.load(f)
        generator.transition = transition
        return generator