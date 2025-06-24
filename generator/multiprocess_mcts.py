import os, logging, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from node import Node

"""
currently not maintained
"""

# top-level def
_transition = None
_reward = None
_filters = None
_max_length = None
_filtered_reward = None

def _init_worker(transition, reward, filters, max_length, filtered_reward):
    # load classes to the memory
    global _transition, _reward, _filters, _max_length, _filtered_reward
    _transition = transition
    _reward = reward
    _filters = filters
    _max_length = max_length
    _filtered_reward = filtered_reward

def _rollout_task(child: Node):
    if child.depth >= _max_length:
        return None 
    return _transition.rollout(child)

def _reward_task(node: Node):
    for f in _filters:
        if not f.check(node):
            return str(node), [-float("inf")], _filtered_reward
    obj_vals, reward = _reward.objective_values_and_reward(node)
    node.clear_cache()
    return str(node), obj_vals, reward

import time
from generator import MCTS

class MultiProcessMCTS(MCTS):
    def __init__(self, *args, n_workers: int=None, mp_context: str="fork", n_rollouts=None, **kwargs):
        """
        n_workers : the number of CPUs (os.cpu_count() if None)
        mp_context: "spawn"|"fork"|"forkserver", must be "spawn" on Windows
        n_rollouts: = n_workers by default
        n_tries: disabled
        """
        self.n_workers  = n_workers or os.cpu_count()
        if n_rollouts is None:
            n_rollouts = n_workers
        super().__init__(*args, n_rollouts=n_rollouts, **kwargs)
        
        self.ctx        = mp.get_context(mp_context)
        self.executor   = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=self.ctx,
            initializer=_init_worker,
            initargs=(self.transition, self.reward, self.filters, self.max_length, self.filtered_reward)
        )

    def _generate_impl(self):
        node = self._selection()
        if node.is_terminal():
            objective_values, reward = self.get_objective_values_and_reward(node)
            if self.terminal_reward != "ignore":
                if self.terminal_reward != "reward":
                    reward = self.terminal_reward
                self._backpropagate(node, reward, False)
            if self.freeze_terminal:
                node.sum_r = -float("inf")
            return
        
        if not node.children:
            self._expand(node)
            
        targets = []
        for _ in range(self.n_rollouts):
            targets.append(node.sample_child())

        # mp rollout
        rollout_futures = [self.executor.submit(_rollout_task, c) for c in targets]
        rollout_results = []

        for fut in as_completed(rollout_futures):
            result_node = fut.result()
            rollout_results.append(result_node)
            
        # record check
        unseen_nodes = []
        for node in rollout_results:
            key = str(node)
            if key not in self.record:
                unseen_nodes.append(node)

        # mp reward
        reward_futures = [self.executor.submit(_reward_task, n) for n in unseen_nodes]
        reward_results = []
        for fut in as_completed(reward_futures):
            key, obj_vals, reward = fut.result()
            if obj_vals[0] != -float("inf"):
                self.log_unique_node(key, obj_vals, reward)
                reward_results.append((key, reward))

        # backpropagate
        reward_map = {k: r for k, r in reward_results}
        for child in targets:
            key = str(child)
            if key in reward_map:
                self._backpropagate(child, reward_map[key], self.use_dummy_reward)
            else:
                self._backpropagate(child, self.filtered_reward, self.use_dummy_reward)

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
            
    # override
    # TODO: make this override unneeded
    def analyze(self):
        self.logger.info("number of generated nodes: " + str(len(self.unique_keys)))
        node_per_sec = len(self.unique_keys) / self.passed_time
        self.logger.info("node_per_sec: " + str(node_per_sec))