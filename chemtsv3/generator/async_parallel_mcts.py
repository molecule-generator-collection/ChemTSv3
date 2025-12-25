import logging
import threading
import time
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from chemtsv3.generator import MCTS
from chemtsv3.node import Node
from chemtsv3.reward import Reward

@dataclass
class RewardTask:
    child: Node
    iters_left: int
    tries_left: int
    unfiltered_flag: bool
    target: Node
    is_direct: bool # direct evaluation / offspring evaluation after rollout
    key: str

@dataclass
class RewardResult:
    task: RewardTask
    objective_values: list
    reward: float

class RewardDispatcher(ABC):
    is_batch_reward_compatible = False
    
    """Abstract dispatcher that accepts reward tasks and yields completed results."""
    def __init__(self, reward: Reward):
        self.reward = reward # can be dummy

    @abstractmethod
    def submit(self, task: RewardTask) -> bool:
        """Submit a task to the dispatcher (becomes inflight). Returns False if inflight is full."""
        raise NotImplementedError

    @abstractmethod
    def pop_ready(self, max_items: int=2**31-1) -> list[RewardResult]:
        """Pop up to max_items completed results. Returns empty list if none."""
        raise NotImplementedError

    @abstractmethod
    def max_inflight(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def inflight(self) -> int:
        raise NotImplementedError

class DummyRewardDispatcher(RewardDispatcher):
    def __init__(self, reward: Reward, max_inflight: int=1, delay_sec: float=2):
        super().__init__(reward=reward)
        if max_inflight <= 0:
            raise ValueError("max_inflight must be >= 1")
        if delay_sec < 0:
            raise ValueError("delay_sec must be >= 0")

        self._max_inflight = max_inflight
        self._delay_sec = delay_sec

        self._pending = queue.Queue() # RewardTask
        self._ready = queue.Queue() # RewardResult

        self._lock = threading.Lock()
        self._inflight = 0
        self._closed = False

        self._worker = threading.Thread(target=self._loop, name="DummyRewardWorker", daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._closed = True

    def submit(self, task: RewardTask) -> bool:
        with self._lock:
            if self._inflight >= self._max_inflight or self._closed:
                return False
            self._inflight += 1
        self._pending.put(task)
        return True

    def pop_ready(self, max_items: int=2**31-1) -> list[RewardResult]:
        out: list[RewardResult] = []
        for _ in range(max_items):
            try:
                out.append(self._ready.get_nowait())
            except queue.Empty:
                break
        return out

    def max_inflight(self) -> int:
        return self._max_inflight

    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def _loop(self) -> None:
        while not self._closed:
            try:
                task = self._pending.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                time.sleep(self._delay_sec)

                objective_values, reward_val = self.reward.objective_values_and_reward(task.target)
                self._ready.put(RewardResult(task=task, objective_values=objective_values, reward=reward_val))
            except Exception as e:
                pass
            finally:
                with self._lock:
                    self._inflight -= 1

class AsyncParallelMCTS(MCTS):
    """
    (WIP) MCTS variant that offloads reward calculation to RewardDispatcher.
    Disabled: failed_parent_reward
    """
    def __init__(self, *args, dispatcher_type: str, max_inflight: int, check_interval: float=0.05, output_dir: str=None, logger: logging.Logger=None, **kwargs):
        super().__init__(*args, output_dir=output_dir, logger=logger, **kwargs) # output_dir and logger are explicit for generator_from_conf()

        self.assign_dispatcher(dispatcher_type, max_inflight, self.reward)
        if not self.dispatcher.is_batch_reward_compatible and self.reward.is_batch_reward():
            raise ValueError("AsyncParallelMCTS requires reward.is_batch_reward() == False with the selected dispatcher.")
        self.check_interval = check_interval # seconds
        
    # override this for custom dispatcher
    # TODO: make this YAML-compatible rather than forcing override
    def assign_dispatcher(self, dispatcher_type: str, max_inflight: int, reward: Reward):
        if dispatcher_type == "dummy":
            self.dispatcher = DummyRewardDispatcher(reward=reward, max_inflight=max_inflight)

    def _generate_impl(self):
        self._drain_ready_results() # harvest all calculated results

        if self.dispatcher.inflight() < self.dispatcher.max_inflight():
            self._fill_queue() # calls _put_reward_task() at last

        if self.dispatcher.inflight() >= self.dispatcher.max_inflight(): # already full
            time.sleep(self.check_interval)
            self._drain_ready_results()

    # override
    def _put_reward_task(self, child):
        self._schedule_one(child, self.n_eval_iters, self.n_tries, False)

    # similar to work_on_queue() / work_on_queue_batch()
    def _schedule_one(self, child: Node, iters: int, tries: int, unfiltered_flag: bool):
        if child.has_reward():
            target = child
            is_direct = True
        else:
            target = self.transition.rollout(child)
            is_direct = False

        pre = self._pre_reward_checks(target)

        if not (type(pre[0]) is bool and pre[0] is True): # no reward calculation
            objective_values, reward = pre
            self.policy.observe(child=child, objective_values=objective_values, reward=reward, is_filtered=(type(objective_values[0])==str))

            if type(objective_values[0]) != str:
                unfiltered_flag = True
                self._backpropagate(child, reward, self.use_dummy_reward)
            else:
                if tries > 1:
                    self._schedule_one(child, iters, tries-1, unfiltered_flag)
                    return
                elif self.filter_reward[int(objective_values[0])] != "ignore":
                    self._backpropagate(child, self.filter_reward[int(objective_values[0])], False)

            if iters > 1:
                self._schedule_one(child, iters-1, self.n_tries, unfiltered_flag)
            elif self.cut_failed_child and not unfiltered_flag:
                child.leave(logger=self.logger)
        else: # reward calculation needed
            key = pre[1]
            self._apply_virtual_loss(child)
            task = RewardTask(child=child, iters_left=iters, tries_left=tries, unfiltered_flag=unfiltered_flag, target=target, is_direct=is_direct, key=key)
            submitted = self.dispatcher.submit(task)
            if submitted:
                self._apply_virtual_loss(child)

    def _drain_ready_results(self):
        results = self.dispatcher.pop_ready()
        if not results:
            return

        for res in results:
            task = res.task
            child = task.child

            self._post_reward_side_effects(task.target, task.key, res.objective_values, res.reward)
            self._revert_virtual_loss(child)

            if task.is_direct and self.reward_cutoff is not None and res.reward < self.reward_cutoff and self.reward_cutoff_warmups < self.n_generated_nodes():
                self.reward_cutoff_count += 1
                child.leave(logger=self.logger)

            self.policy.observe(child=child, objective_values=res.objective_values, reward=res.reward, is_filtered=False)

            task.unfiltered_flag = True
            self._backpropagate(child, res.reward, self.use_dummy_reward)

            if task.iters_left > 1:
                self._schedule_one(child, task.iters_left - 1, self.n_tries, task.unfiltered_flag)