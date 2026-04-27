import importlib
import logging
import os
import threading
import time
from typing import Any
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass

from mpi4py import MPI

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
    worker_rank: int = None
    worker_local_index: int = None

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

TAG_TASK = 1
TAG_RESULT = 2
TAG_STOP = 3

@dataclass
class _InflightTaskState:
    """Rank-0 only local state for matching worker results back to original tasks."""
    task: "RewardTask"
    worker_rank: int
    worker_local_index: int

class MPIRewardDispatcher(RewardDispatcher):
    is_batch_reward_compatible = False

    def __init__(self, reward, comm=None, max_inflight=None):
        super().__init__(reward=reward)

        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.rank != 0: # rank 0 ... master / rank 1~ ... workers
            raise ValueError("MPIRewardDispatcher must be instantiated on rank 0.")

        if self.size < 2:
            raise ValueError("MPIRewardDispatcher requires at least 2 MPI ranks.")

        self._ready = queue.Queue()
        self._closed = False

        self._idle_workers = set(range(1, self.size))
        self._inflight_tasks: dict[int, _InflightTaskState] = {}
        self._next_task_id = 0
        self._worker_task_counts = {rank: 0 for rank in range(1, self.size)}

        self._max_inflight = self.size - 1 if max_inflight is None else max_inflight
        if self._max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        self._max_inflight = min(self._max_inflight, self.size - 1)

    def submit(self, task: "RewardTask") -> bool:
        if self._closed:
            return False

        self._poll_results()

        if len(self._inflight_tasks) >= self._max_inflight:
            return False

        if not self._idle_workers:
            return False

        worker_rank = self._idle_workers.pop()
        task_id = self._next_task_id
        self._next_task_id += 1
        worker_local_index = self._worker_task_counts[worker_rank]
        self._worker_task_counts[worker_rank] += 1

        node = task.target
        if not hasattr(node, "pack"):
            raise TypeError(f"{node.__class__.__name__} must implement pack() for MPIRewardDispatcher.")

        payload = {
            "task_id": task_id,
            "node_module_name": node.__class__.__module__,
            "node_class_name": node.__class__.__name__,
            "node_payload": node.pack(),
        }

        self.comm.send(payload, dest=worker_rank, tag=TAG_TASK)
        self._inflight_tasks[task_id] = _InflightTaskState(task=task, worker_rank=worker_rank, worker_local_index=worker_local_index)
        return True

    def pop_ready(self, max_items: int = 2**31 - 1) -> list["RewardResult"]:
        self._poll_results()

        out = []
        for _ in range(max_items):
            try:
                out.append(self._ready.get_nowait())
            except queue.Empty:
                break
        return out

    def max_inflight(self) -> int:
        return self._max_inflight

    def inflight(self) -> int:
        self._poll_results()
        return len(self._inflight_tasks)

    def close(self) -> None:
        """Tell all workers to stop. Called after generation finishes on rank 0."""
        if self._closed:
            return

        self._poll_results()

        for worker_rank in range(1, self.size):
            self.comm.send(None, dest=worker_rank, tag=TAG_STOP)

        self._closed = True

    def _poll_results(self) -> None:
        """Non-blocking polling for completed worker results."""
        status = MPI.Status()

        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status):
            src = status.Get_source()
            payload = self.comm.recv(source=src, tag=TAG_RESULT)

            task_id = payload["task_id"]
            inflight = self._inflight_tasks.pop(task_id, None)

            # If something got mismatched, skip safely.
            if inflight is None:
                self._idle_workers.add(src)
                continue

            self._idle_workers.add(src)
            task = inflight.task

            if payload["status"] == "ok":
                self._ready.put(
                    RewardResult(
                        task=task, 
                        objective_values=payload["objective_values"], 
                        reward=payload["reward"],
                        worker_rank=inflight.worker_rank, 
                        worker_local_index=inflight.worker_local_index
                    )
                )
            else:
                self._ready.put(RewardResult(
                        task=task,
                        objective_values=[payload.get("error_code", "mpi_reward_error")],
                        reward=0.0,
                        worker_rank=inflight.worker_rank,
                        worker_local_index=inflight.worker_local_index
                    )
                )


def reconstruct_node(node_module_name: str, node_class_name: str, node_payload: Any):
    module = importlib.import_module(node_module_name)
    cls = getattr(module, node_class_name)

    if not hasattr(cls, "unpack"):
        raise TypeError(f"{node_module_name}.{node_class_name} must implement unpack().")

    return cls.unpack(node_payload)

def worker_loop(reward, comm=None, logger=None) -> None:
    comm = comm or MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        raise ValueError("worker_loop must not run on rank 0.")

    status = MPI.Status()

    while True:
        payload = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP:
            break

        if tag != TAG_TASK:
            continue

        task_id = payload["task_id"]

        try:
            node = reconstruct_node(
                payload["node_module_name"],
                payload["node_class_name"],
                payload["node_payload"],
            )
            objective_values, reward_value = reward.objective_values_and_reward(node)
            result = {
                "task_id": task_id,
                "status": "ok",
                "objective_values": objective_values,
                "reward": reward_value,
            }
        except Exception as e:
            if logger is not None:
                logger.exception("Reward evaluation failed on worker rank %d", rank)
            else:
                import traceback
                print(f"[worker {rank}] reward evaluation failed: {e!r}", flush=True)
                traceback.print_exc()

            result = {
                "task_id": task_id,
                "status": "error",
                "error": repr(e),
                "error_code": "mpi_reward_error",
            }

        comm.send(result, dest=0, tag=TAG_RESULT)

class AsyncParallelMCTS(MCTS):
    """
    MCTS variant that offloads reward calculation to RewardDispatcher.
    Disabled: failed_parent_reward
    """
    def __init__(self, *args, max_inflight: int, dispatcher_type: str=None, check_interval: float=0.05, output_dir: str=None, logger: logging.Logger=None, **kwargs):
        super().__init__(*args, output_dir=output_dir, logger=logger, **kwargs) # output_dir and logger are explicit for generator_from_conf()

        self.assign_dispatcher(dispatcher_type, max_inflight, self.reward)
        if self.dispatcher is not None:
            if not self.dispatcher.is_batch_reward_compatible and self.reward.is_batch_reward():
                raise ValueError("AsyncParallelMCTS requires reward.is_batch_reward() == False with the selected dispatcher.")
        self.check_interval = check_interval # seconds
        self._pending_generation_meta = {}
        self._generation_trace_path = os.path.join(self.output_dir(), "results_with_local_ids.tsv")
        if not os.path.exists(self._generation_trace_path):
            with open(self._generation_trace_path, "w") as f:
                f.write("generation_id\tworker_rank\tworker_local_index\tkey\treward\tobjective_values\n")
        
    # override this for custom dispatcher
    # TODO: make this YAML-compatible rather than forcing override
    def assign_dispatcher(self, dispatcher_type: str, max_inflight: int, reward):
        if dispatcher_type == "dummy":
            self.dispatcher = DummyRewardDispatcher(reward=reward, max_inflight=max_inflight)
        elif dispatcher_type == "mpi":
            self.dispatcher = MPIRewardDispatcher(reward=reward, max_inflight=max_inflight)
        elif dispatcher_type is None:
            self.dispatcher = None
        else:
            raise ValueError(f"Unknown dispatcher_type: {dispatcher_type}")

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
            self._pending_generation_meta[task.key] = (res.worker_rank, res.worker_local_index)

            self._post_reward_side_effects(task.target, task.key, res.objective_values, res.reward)
            self._record_results_with_local_ids(task.key, res.objective_values, res.reward)
            self._revert_virtual_loss(child)

            if task.is_direct and self.reward_cutoff is not None and res.reward < self.reward_cutoff and self.reward_cutoff_warmups < self.n_generated_nodes():
                self.reward_cutoff_count += 1
                child.leave(logger=self.logger)

            self.policy.observe(child=child, objective_values=res.objective_values, reward=res.reward, is_filtered=False)

            task.unfiltered_flag = True
            self._backpropagate(child, res.reward, self.use_dummy_reward)

            if task.iters_left > 1:
                self._schedule_one(child, task.iters_left - 1, self.n_tries, task.unfiltered_flag)
                
    # record results with local ids
    def _record_results_with_local_ids(self, key: str, objective_values: list[float], reward: float):
        meta = self._pending_generation_meta.pop(key, None)
        if meta is None:
            return

        worker_rank, worker_local_index = meta
        generation_id = self.record[key]["generation_order"] - 1
            
        obj_str = ",".join(map(str, objective_values))
        with open(self._generation_trace_path, "a") as f:
            f.write(f"{generation_id}\tworker_{worker_rank}\t{worker_local_index}\t{key}\t{reward}\t{obj_str}\n")