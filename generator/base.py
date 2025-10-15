from abc import ABC, abstractmethod
import copy
from datetime import datetime
import logging
import math
import os
import pickle
import queue
import time
from typing import Self
import yaml
import matplotlib.pyplot as plt
import numpy as np
from filter import Filter
from node import Node
from reward import Reward, LogPReward
from transition import Transition
from utils import moving_average, log_memory_usage, make_logger, flush_delayed_logger, is_running_under_slurm, make_subdirectory, plot_xy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

class Generator(ABC):
    """Base generator class. Override _generate_impl (and __init__) to implement."""
    def __init__(self, transition: Transition, reward: Reward=LogPReward(), filters: list[Filter]=None, filter_reward: float | str | list=0, name: str=None, output_dir: str=None, logger: logging.Logger=None, logging_interval: int=None, info_interval: int=100, analyze_interval: int=10000, verbose_interval: int=None, save_interval: int=None, save_on_completion: bool=False, include_transition_to_save: bool=False):
        """
        Args:
            filter_reward: Substitute reward value used when nodes are filtered. Set to "ignore" to skip reward assignment. Use a list to specify different rewards for each filter step.
            
            output_dir: Directory where the generation results and logs will be saved.
            logger: Logger instance used to record generation results.
            logging_interval: Number of generations between each logging. Overrides info_interval.
            info_interval: Number of generations between each logging of the generation result.
            save_interval: Number of generations between each checkpoint save.
            save_on_completion: If True, saves the checkpoint when completing the generation.
            include_transition_to_save: If True, transition will be included to the checkpoint file when saving.
        """
        self.transition = transition
        self._name = name or self._make_name()
        self.reward: Reward = reward
        self.filters: list[Filter] = filters or []
        if type(filter_reward) != list:
            self.filter_reward = [filter_reward for _ in range(len(self.filters))]
        else:
            if len(filters) != len(filter_reward):
                raise ValueError(f"Size mismatch: 'filters': {len(filters)}, 'filter_reward': {len(filters)}")
            self.filter_reward = filter_reward
        self.filter_counts = [0] * len(self.filters)
        self._output_dir = output_dir or make_subdirectory(os.path.join(REPO_ROOT, "sandbox", "generation_result"))
        os.makedirs(self.output_dir(), exist_ok=True)
        self.unique_keys = []
        self.record: dict[str, dict] = {} # save at least all of the following for unique keys: "objective_values", "reward", "generation_order", "time"
        self.best_reward = -float("inf")
        self.worst_reward = float("inf")
        self.passed_time = 0
        self.grab_count = 0
        self.duplicate_count = 0
        self.logger = logger or make_logger(output_dir=self.output_dir(), name=self.name())
        self.yaml_copy = None
        if logging_interval is None:
            if is_running_under_slurm():
                self.logger.info("Slurm detected. Setting logging_interval to 1000 to avoid I/O overhead. Specify logging_interval to override this behavior.")
                self.logging_interval = 1000
            else:
                self.logging_interval = 1
        else:
            self.logging_interval = logging_interval
        self.info_interval = info_interval
        self.analyze_interval = analyze_interval
        self.verbose_interval = verbose_interval
        self.save_interval = save_interval
        self.save_on_completion = save_on_completion
        self.include_transition_to_save = include_transition_to_save
        self.last_saved = 0
        self.next_save = save_interval
        self.initial_time, self.time_start = 0, 0 # precaution
    
    @abstractmethod
    def _generate_impl(self, *kwargs):
        pass

    def generate(self, max_generations: int=None, time_limit: float=None):
        """
        Generate nodes that either is_terminal() = True or depth = max_length. Tries to maximize the reward by MCTS search.

        Args:
            max_generations: Generation stops after generating 'max_generations' number of nodes.
            time_limit: Seconds. Generation stops after the time limit.
        """
        if (time_limit is None) and (max_generations is None):
            raise ValueError("Specify at least one of max_genrations, max_rollouts or time_limit.")
        
        if self.verbose_interval is not None:
            self.log_verbose_info()
        
        # record current time and counts
        self.time_start = time.time()
        self.initial_time = self.passed_time
        if self.passed_time == 0:
            self._write_csv_header()
        initial_count_generations = len(self.unique_keys)
        
        self.logger.info("Starting generation...")
        try:
            while True:
                time_passed = time.time() - self.time_start
                self.passed_time = self.initial_time + time_passed
                if time_limit is not None and time_passed >= time_limit:
                    break
                if max_generations is not None and len(self.unique_keys) - initial_count_generations >= max_generations:
                    break
                if self.should_finish():
                    break
                self._generate_impl()
                
                if self.save_interval is not None and (self.n_generated_nodes() >= self.next_save):
                    self.save()
        except KeyboardInterrupt:
            self.logger.warning("Generation interrupted by user (KeyboardInterrupt).")
        except SystemExit:
            pass
        except Exception as e:
            self.logger.exception(f"Unexpected error occurred: {e}")
        finally:
            if hasattr(self, "executor"): # for MP
                self.executor.shutdown(cancel_futures=True)
                self.logger.info("Executor shutdown completed.")
            self.logger.info("Generation finished.")
            
            if (self.save_interval is not None or self.save_on_completion) and self.n_generated_nodes() != self.last_saved:
                self.save(is_interval=False)
                
    def should_finish(self):
        return False

    def _make_name(self):
        return datetime.now().strftime("%m-%d_%H-%M") + "_" + self.__class__.__name__
    
    def name(self):
        return self._name
    
    def output_dir(self):
        return self._output_dir if self._output_dir.endswith(os.sep) else self._output_dir + os.sep

    def _write_csv_header(self):
        header = ["order", "time", "key", "reward"]
        if not self.reward.is_single_objective:
            header += [f.__name__ for f in self.reward.objective_functions()]
        self.logger.info(header)

    def _log_unique_node(self, key, objective_values, reward):
        self.unique_keys.append(key)
        self.record[key] = {}
        self.record[key]["objective_values"] = objective_values
        self.record[key]["reward"] = reward
        self.passed_time = self.initial_time + (time.time() - self.time_start)
        self.record[key]["time"] = self.passed_time
        self.record[key]["generation_order"] = len(self.unique_keys)
        
        self.worst_reward = min(self.worst_reward, reward)
        
        if self.info_interval <= 1 or reward > self.best_reward:
            if reward > self.best_reward:
                self.best_reward = reward
                prefix = "<Best reward updated> "
            else:
                prefix = ""
            self.logger.info(prefix + str(len(self.unique_keys)) + " - time: " + "{:.2f}".format(self.passed_time) + ", reward: " + "{:.4f}".format(reward) + ", node: " + key)
        else:
            if len(self.unique_keys)%self.info_interval == 0:
                average = self.average_reward(self.info_interval)
                self.logger.info(str(len(self.unique_keys)) + " - time: " + "{:.2f}".format(self.passed_time) + ", average over " + str(self.info_interval) + ": " + "{:.4f}".format(average))

        if self.reward.is_single_objective:
            row = [len(self.unique_keys), self.passed_time, key, reward]
        else:
            row = [len(self.unique_keys), self.passed_time, key, reward, *objective_values]
            
        self.logger.info(row)
        
        if self.analyze_interval is not None and len(self.unique_keys) % self.analyze_interval == 0:
            self.analyze()
        if self.logging_interval is not None and len(self.unique_keys) % self.logging_interval == 0:
            flush_delayed_logger(self.logger)
        if self.verbose_interval is not None and len(self.unique_keys) % self.verbose_interval == 0:
            self.log_verbose_info()
        
    def average_reward(self, window: int | float=None, top_p: float = None):
        """
        Compute the average reward over the most recent `window` entries.
        If `top_p` is specified (e.g., 0.1), return the average of the top `top_p * 100%` rewards within that window.

        Args:
            window: Number or rate of recent entries to consider.
            top_p: Fraction of top rewards to average, e.g., 0.1 means top 10%.
        """
        if window is None:
            window = len(self.unique_keys)
        elif window < 1:
            window = max(1, math.floor(window * len(self.unique_keys)))
        window = min(window, len(self.unique_keys))
        rewards = [self.record[k]["reward"] for k in self.unique_keys[-window:]]
        
        if top_p is not None:
            if top_p <= 0 or 1 <= top_p:
                self.logger.warning("'top_p' is ignored as it is not within (0, 1).")
            else:
                rewards = sorted(rewards, reverse=True)
                top_k = max(1, math.floor(len(rewards) * top_p))
                rewards = rewards[:top_k]
        return np.average(rewards)

    def _get_objective_values_and_reward(self, node: Node) -> tuple[list[float], float]:
        self.grab_count += 1
        key = node.key()
        if key in self.record:
            self.duplicate_count += 1
            self.logger.debug("Already in dict: " + key + ", reward: " + str(self.record[key]["reward"]))
            node.clear_cache()
            return self.record[key]["objective_values"], self.record[key]["reward"]
        
        for i, filter in enumerate(self.filters):
            filter_result = filter.check(node)
            if type(filter_result) in (float, int) or filter_result == False:
                self.filter_counts[i] += 1
                if type(filter_result) in (float, int):
                    filter_reward = filter_result
                    self.logger.debug(f"Filtered by {filter.__class__.__name__}: {key}, reward override={filter_result}")
                elif filter_result == False:
                    filter_reward = self.filter_reward[i]
                    self.logger.debug(f"Filtered by {filter.__class__.__name__}: {key}")
                
                self.transition.observe(node=node, objective_values=[str(i)], reward=filter_reward, is_filtered=True)
                for filter in self.filters:
                    filter.observe(node=node, objective_values=[str(i)], reward=filter_reward, is_filtered=True)
                    
                node.clear_cache()
                return [str(i)], self.filter_reward[i]
            
        objective_values, reward = self.reward.objective_values_and_reward(node)
        
        self._log_unique_node(key, objective_values, reward)
        
        node.reward = reward
        self.transition.observe(node=node, objective_values=objective_values, reward=reward, is_filtered=False)
        for filter in self.filters:
            filter.observe(node=node, objective_values=objective_values, reward=reward, is_filtered=False)

        self.on_generation(node, objective_values=objective_values, reward=reward)
        
        node.clear_cache()
        return objective_values, reward
    
    def on_generation(self, node: Node, objective_values: list[float], reward: float):
        pass

    # visualize results
    def plot(self, x_axis: str="generation_order", moving_average_window: int | float=0.01, max_curve=True, max_line=False, xlim: tuple[float, float]=None, ylims: dict[str, tuple[float, float]]=None, linewidth: float=1.0, packed_objectives=None, save_only: bool=False, reward_top_ps: list[float]=None):
        if len(self.unique_keys) == 0:
            return
        self._plot_objective_values_and_reward(x_axis=x_axis, moving_average_window=moving_average_window, max_curve=max_curve, max_line=max_line, xlim=xlim, ylims=ylims, linewidth=linewidth, save_only=save_only, reward_top_ps=reward_top_ps)
        if packed_objectives:
            for po in packed_objectives:
                self._plot_specified_objective_values(po, x_axis=x_axis, moving_average_window=moving_average_window, xlim=xlim, linewidth=linewidth, save_only=save_only)

    def _plot(self, x_axis: str="generation_order", y_axis: str | list[str]="reward", moving_average_window: int | float=0.01, max_curve=True, max_line=False, scatter=True, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, loc: str="lower right", linewidth: float=1.0, save_only: bool=False, top_ps: list[float]=None):
        top_ps = top_ps or []
        x = [self.record[molkey][x_axis] for molkey in self.unique_keys]

        reward_name = self.reward.name()
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
        
        plot_xy(x, y, x_axis=x_axis, y_axis=y_axis, moving_average_window=moving_average_window, max_curve=max_curve, max_line=max_line, scatter=scatter, xlim=xlim, ylim=ylim, loc=loc, linewidth=linewidth, save_only=save_only, top_ps=top_ps, output_dir=self.output_dir(), title=self.name(), logger=self.logger)

    def _plot_objective_values_and_reward(self, x_axis: str="generation_order", moving_average_window: int | float=0.01, max_curve=True, max_line=False, xlim: tuple[float, float]=None, ylims: dict[str, tuple[float, float]]=None, loc: str="lower right", linewidth: float=0.01, save_only: bool=False, reward_top_ps: list[float]=None):
        ylims = ylims or {}
        if not self.reward.is_single_objective:
            objective_names = [f.__name__ for f in self.reward.objective_functions()]
            for o in objective_names:
                self._plot(x_axis=x_axis, y_axis=o, moving_average_window=moving_average_window, max_curve=False, max_line=False, xlim=xlim, ylim=ylims.get(o, None), linewidth=linewidth, save_only=save_only)
        self._plot(x_axis=x_axis, y_axis="reward", moving_average_window=moving_average_window, max_curve=max_curve, max_line=max_line, xlim=xlim, ylim=ylims.get("reward", None), loc=loc, linewidth=linewidth, save_only=save_only, top_ps=reward_top_ps)

    def _plot_specified_objective_values(self, y_axes: list[str], x_axis: str="generation_order", moving_average_window: int | float=0.01, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, linewidth: float=1.0, save_only: bool=False):
        x = [self.record[molkey][x_axis] for molkey in self.unique_keys]
        objective_names = [f.__name__ for f in self.reward.objective_functions()]
        for ya in y_axes:
            label = ya
            objective_idx = objective_names.index(ya)
            y = [self.record[molkey]["objective_values"][objective_idx] for molkey in self.unique_keys]
            y_ma = moving_average(y, moving_average_window)
            plt.plot(x, y_ma, label=label, linewidth=linewidth)
        plt.grid(axis="y")
        plt.title(self.name())
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.savefig(self.output_dir() + self.name() + "_by_" + x_axis + ".png", bbox_inches="tight")
        plt.close() if save_only else plt.show()

    def analyze(self):
        if len(self.unique_keys) == 0:
            return
        self.logger.info(f"Generation count: {self.n_generated_nodes()}")
        self.logger.info(f"Unique rate: {self.unique_rate():.3f}")
        self.logger.info(f"Node per sec: {self.node_per_sec():.3f}")
        self.logger.info(f"Best reward: {self.best_reward:.3f}")
        self.logger.info(f"Average reward: {self.average_reward():.3f}")
        top_10_auc = self.auc(top_k=10)
        if self.worst_reward >= 0 and self.best_reward <= 1:
            self.logger.info(f"Top 10 AUC: {top_10_auc:.3f}")
        if len(self.filters) != 0:
            self.logger.info(f"Filter counts (reward): {self.filter_counts}")
        self.transition.analyze()
        for filter in self.filters:
            filter.analyze()
        self.reward.analyze()
        
    def display_top_k_molecules(self, str2mol_func, k: int=15, mols_per_row=5, legends: list[str]=["order","reward"], target: str="reward", size=(200, 200)):
        from utils import draw_mols, top_k_df
        df = self.df()
        top_k = top_k_df(df, k=k, target=target)
        draw_mols(top_k, legends=legends, mols_per_row=mols_per_row, size=size, str2mol_func=str2mol_func)
    
    def n_generated_nodes(self):
        return len(self.unique_keys)
        
    def valid_rate(self):
        return 1 - (sum(self.filter_counts) / self.grab_count)
    
    def unique_rate(self):
        return 1 - (self.duplicate_count / self.grab_count)
    
    def node_per_sec(self):
        return len(self.unique_keys) / self.passed_time
        
    def auc(self, top_k: int=10, max_oracle_calls: int=None, freq_log: int=100, finish: bool=False):
        """
        Returns the AUC of the average of the top-k rewards. Assumes all rewards lie within the range [0, 1].
        Ref: https://github.com/wenhao-gao/mol_opt/blob/2da631be85af8d10a2bb43f2de76a03171166190/main/optimizer.py#L30
        """
        max_oracle_calls = max_oracle_calls or self.n_generated_nodes()
        buffer = {
            key: (self.record[key]["reward"], self.record[key]["generation_order"])
            for key in self.unique_keys
        }

        sum_auc = 0
        top_k_mean_prev = 0
        called = 0

        ordered_results = sorted(buffer.items(), key=lambda kv: kv[1][1])

        for idx in range(freq_log, min(len(ordered_results), max_oracle_calls), freq_log):
            temp = ordered_results[:idx]
            top_k_results = sorted(temp, key=lambda kv: kv[1][0], reverse=True)[:top_k]
            top_k_mean = np.mean([x[1][0] for x in top_k_results])
            sum_auc += freq_log * (top_k_mean + top_k_mean_prev) / 2
            top_k_mean_prev = top_k_mean
            called = idx
            
        top_k_results = sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True)[:top_k]
        top_k_mean = np.mean([x[1][0] for x in top_k_results])
        sum_auc += (len(ordered_results) - called) * (top_k_mean + top_k_mean_prev) / 2

        if finish and len(ordered_results) < max_oracle_calls:
            sum_auc += (max_oracle_calls - len(ordered_results)) * top_k_mean

        return sum_auc / max_oracle_calls
    
    def top_k(self, k: int = 1) -> list[tuple[str, float]]:
        key_rewards = [(key, self.record[key]["reward"]) for key in self.unique_keys]
        key_index = {key: idx for idx, key in enumerate(self.unique_keys)} # tiebreaker
        key_rewards.sort(key=lambda x: (x[1], key_index[x[0]]), reverse=True)
        return key_rewards[:k]
        
    def generated_keys(self, last: int=None) -> list[str]:
        if last is None:
            return self.unique_keys
        else:
            return self.unique_keys[-last:]
        
    def df(self, resort: bool=False):
        """
        Return a pandas.DataFrame with the same columns/rows as the CSV logging.
        Columns:
            order, time, key, <reward_name>[, <objective_1>, <objective_2>, ...]
        Rows are ordered by generation_order.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas not found."
            ) from e

        if len(self.unique_keys) == 0:
            cols = ["order", "time", "key", "reward"]
            if not self.reward.is_single_objective:
                cols += [f.__name__ for f in self.reward.objective_functions()]
            return pd.DataFrame(columns=cols)

        columns = ["order", "time", "key", "reward"]
        if not self.reward.is_single_objective:
            columns += [f.__name__ for f in self.reward.objective_functions()]

        rows = []
        for key in self.unique_keys:
            rec = self.record[key]
            base = [rec["generation_order"], rec["time"], key, rec["reward"]]
            if not self.reward.is_single_objective:
                base += rec["objective_values"]
            rows.append(base)

        df = pd.DataFrame(rows, columns=columns)

        # to numeric
        for col in ("order", "time", "reward"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if not self.reward.is_single_objective:
            for fn in self.reward.objective_functions():
                name = fn.__name__
                if name in df.columns:
                    df[name] = pd.to_numeric(df[name], errors="coerce")

        # shouldn't be needed for the default generators
        if resort:
            df.sort_values("order", inplace=True, kind="stable")
            df.reset_index(drop=True, inplace=True)
            
        return df
        
    def inherit(self, predecessor: Self):
        self._output_dir = predecessor._output_dir
        self.logger = predecessor.logger
        self.record = predecessor.record
        self.best_reward = predecessor.best_reward
        self.unique_keys = predecessor.unique_keys
        self.passed_time = predecessor.passed_time
        if self.save_interval is not None:
            self.last_saved = predecessor.last_saved
            self.next_save = self.n_generated_nodes() + self.next_save
        
    def log_verbose_info(self):
        log_memory_usage(self.logger)
        
    def __getstate__(self):
        state = self.__dict__.copy()
        if "transition" in state and not self.include_transition_to_save:
            del state["transition"]
        # make queue picklable (for MCTS)
        rq = state.get("reward_queue", None)
        if isinstance(rq, queue.Queue):
            with rq.mutex:
                state["reward_queue"] = list(rq.queue)
        return state
    
    def __setstate__(self, state):
        # rebuild queue (for MCTS)
        saved_queue = state.get("reward_queue", None)
        if isinstance(saved_queue, list):
            restored_queue = queue.Queue()
            for item in saved_queue:
                restored_queue.put(item)
            state["reward_queue"] = restored_queue
        self.__dict__.update(state)
    
    def _set_yaml_copy(self, conf: dict):
        self.yaml_copy = copy.deepcopy(conf)
    
    def save(self, is_interval=True):
        for ha in self.logger.handlers:
            if isinstance(ha, logging.FileHandler):
                log_dir = os.path.dirname(ha.baseFilename)  
                log_file_without_ext = os.path.splitext(os.path.basename(ha.baseFilename))[0]
                self._log_dir = log_dir
                self._log_file = log_file_without_ext
                self._file_level = ha.level
            elif isinstance(ha, logging.StreamHandler):
                self._console_level = ha.level
            else:
                self._csv_level = ha.level
        if is_interval and self.save_interval is not None:
            self.next_save += self.save_interval
            self.last_saved = self.n_generated_nodes()
        save_dir = os.path.join(self.output_dir(), "checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        if self.yaml_copy is not None:
            path = os.path.join(save_dir, "config.yaml")
            with open(path, "w") as f:
                yaml.dump(self.yaml_copy, f, sort_keys=False)
        with open(os.path.join(save_dir, "checkpoint.gtr"), mode="wb") as fo:
            pickle.dump(self, fo)
        self.logger.info(f"Checkpoint saved at {self.n_generated_nodes()} generations.")

    def load_file(file: str, transition: Transition=None) -> Self:
        with open(file, "rb") as f:
            generator = pickle.load(f)
        generator.logger.warning(f"Logs will be written to: {generator._log_dir} instead of newly created one.")
        generator.logger = make_logger(output_dir=generator._log_dir, name=generator._log_file, console_level=generator._console_level, file_level=generator._file_level, csv_level=generator._csv_level)
        if transition is None and not hasattr(generator, "transition"):
            raise ValueError("Transition is not specified in load_file(), nor saved in the checkpoint.")
        elif transition is not None and not hasattr(generator, "transition"):
            generator.transition = transition
        else:
            generator.logger.info(f"Loading transition from the checkpoint.")
        return generator
    
    def load_dir(dir: str) -> Self:
        from utils import conf_from_yaml, generator_from_conf
        conf = conf_from_yaml(os.path.join(dir, "config.yaml"))
        transition = generator_from_conf(conf).transition
        return Generator.load_file(os.path.join(REPO_ROOT, dir, "checkpoint.gtr"), transition)