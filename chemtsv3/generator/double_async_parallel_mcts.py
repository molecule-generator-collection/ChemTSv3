import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from chemtsv3.generator.async_parallel_mcts import (
    AsyncParallelMCTS,
    MPIRewardDispatcher,
    MPIWorkerPool,
    RewardTask,
    pack_node_for_mpi,
    reconstruct_node,
)
from chemtsv3.node import Node
from chemtsv3.transition import Transition

@dataclass
class TransitionTask:
    kind: str # "expand", "rollout"
    node: Node
    iters_left: int = None
    tries_left: int = None
    unfiltered_flag: bool = False

@dataclass
class TransitionResult:
    task: TransitionTask
    nexts: list[Node] = None
    target: Node = None
    error: Exception = None

class TransitionDispatcher(ABC):
    """Abstract dispatcher that accepts transition tasks and yields completed results."""
    def __init__(self, transition: Transition):
        self.transition = transition

    @abstractmethod
    def submit(self, task: TransitionTask) -> bool:
        """Submit a task to the dispatcher. Returns False if inflight is full."""
        raise NotImplementedError

    @abstractmethod
    def pop_ready(self, max_items: int=2**31-1) -> list[TransitionResult]:
        raise NotImplementedError

    @abstractmethod
    def max_inflight(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def inflight(self) -> int:
        raise NotImplementedError

class ThreadedTransitionDispatcher(TransitionDispatcher):
    def __init__(self, transition: Transition, max_inflight: int=1):
        super().__init__(transition=transition)
        if max_inflight <= 0:
            raise ValueError("max_inflight must be >= 1")

        self._max_inflight = max_inflight
        self._pending = queue.Queue()
        self._ready = queue.Queue()
        self._lock = threading.Lock()
        self._inflight = 0
        self._closed = False

        self._workers = []
        for i in range(max_inflight):
            worker = threading.Thread(target=self._loop, name=f"TransitionWorker-{i}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def close(self) -> None:
        self._closed = True

    def submit(self, task: TransitionTask) -> bool:
        with self._lock:
            if self._inflight >= self._max_inflight or self._closed:
                return False
            self._inflight += 1
        self._pending.put(task)
        return True

    def pop_ready(self, max_items: int=2**31-1) -> list[TransitionResult]:
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
        with self._lock:
            return self._inflight

    def _loop(self) -> None:
        while not self._closed:
            try:
                task = self._pending.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                if task.kind == "expand":
                    self._ready.put(TransitionResult(task=task, nexts=self.transition.next_nodes(task.node)))
                elif task.kind == "rollout":
                    self._ready.put(TransitionResult(task=task, target=self.transition.rollout(task.node)))
                else:
                    raise ValueError(f"Unknown transition task kind: {task.kind}")
            except Exception as e:
                self._ready.put(TransitionResult(task=task, error=e))
            finally:
                with self._lock:
                    self._inflight -= 1

class MPITransitionDispatcher(TransitionDispatcher):
    def __init__(self, transition: Transition, pool: MPIWorkerPool):
        super().__init__(transition=transition)
        self.pool = pool

    def submit(self, task: TransitionTask) -> bool:
        payload = pack_node_for_mpi(task.node)
        if task.kind == "expand":
            return self.pool.submit("transition_expand", payload, task)
        if task.kind == "rollout":
            return self.pool.submit("transition_rollout", payload, task)
        raise ValueError(f"Unknown transition task kind: {task.kind}")

    def pop_ready(self, max_items: int=2**31-1) -> list[TransitionResult]:
        out = []

        for res in self.pool.pop_ready("transition_expand", max_items=max_items):
            task = res.task
            payload = res.payload
            if payload["status"] == "ok":
                nexts = [reconstruct_node(p["node_module_name"], p["node_class_name"], p["node_payload"]) for p in payload["nodes"]]
                out.append(TransitionResult(task=task, nexts=nexts))
            else:
                out.append(TransitionResult(task=task, error=RuntimeError(payload.get("error", "mpi_transition_expand_error"))))

        remaining = max_items - len(out)
        if remaining <= 0:
            return out

        for res in self.pool.pop_ready("transition_rollout", max_items=remaining):
            task = res.task
            payload = res.payload
            if payload["status"] == "ok":
                p = payload["node"]
                target = reconstruct_node(p["node_module_name"], p["node_class_name"], p["node_payload"])
                out.append(TransitionResult(task=task, target=target))
            else:
                out.append(TransitionResult(task=task, error=RuntimeError(payload.get("error", "mpi_transition_rollout_error"))))

        return out

    def max_inflight(self) -> int:
        return self.pool.max_inflight()

    def inflight(self) -> int:
        return self.pool.inflight()

class DoubleAsyncParallelMCTS(AsyncParallelMCTS):
    """
    AsyncParallelMCTS variant that also parallelize transition (both expansion and rollout). For transition-only parallel search, set 'reward_dispatcher_type' to 'disable'.
    """
    def __init__(self, *args, inflight_type: str="separate", 
        max_mpi_inflight: int=None, # For inflight_type="mpi"
        max_reward_inflight: int=None, reward_dispatcher_type: str=None,
        max_transition_inflight: int=None, transition_dispatcher_type: str=None,
        transition_loss: float=0.0, # Virtual loss for transition
        output_dir: str=None, logger: logging.Logger=None, **kwargs):
        
        if inflight_type == "mpi":
            reward_dispatcher_type = reward_dispatcher_type or "mpi"
            transition_dispatcher_type = transition_dispatcher_type or "mpi"
            if reward_dispatcher_type not in ("mpi", "disable"):
                raise ValueError("inflight_type='mpi' supports reward_dispatcher_type='mpi' or 'disable'.")
            if transition_dispatcher_type != "mpi":
                raise ValueError("inflight_type='mpi' requires transition_dispatcher_type='mpi'.")
        else:
            transition_dispatcher_type = transition_dispatcher_type or "thread"

        self.reward_dispatcher_type = reward_dispatcher_type
        dispatcher_arg = None if inflight_type == "mpi" or reward_dispatcher_type in (None, "disable") else reward_dispatcher_type
        reward_max_inflight, transition_max_inflight = self._resolve_inflight_limits(
            inflight_type=inflight_type,
            max_reward_inflight=max_reward_inflight,
            max_transition_inflight=max_transition_inflight,
            max_mpi_inflight=max_mpi_inflight,
        )
        self.inflight_type = inflight_type
        self.max_mpi_inflight = max_mpi_inflight

        super().__init__(*args, max_inflight=reward_max_inflight, reward_dispatcher_type=dispatcher_arg, output_dir=output_dir, logger=logger, **kwargs)
        self.mpi_worker_pool = None
        if self.inflight_type == "mpi":
            self.mpi_worker_pool = MPIWorkerPool(max_inflight=max_mpi_inflight)
            if self.reward_dispatcher_type == "mpi":
                self.dispatcher = MPIRewardDispatcher(reward=self.reward, pool=self.mpi_worker_pool)
                if not os.path.exists(self._generation_trace_path):
                    with open(self._generation_trace_path, "w") as f:
                        f.write("generation_id\tworker_rank\tworker_local_index\tkey\treward\tobjective_values\n")

        self.transition_loss = transition_loss
        self.assign_transition_dispatcher(
            transition_dispatcher_type=transition_dispatcher_type,
            max_transition_inflight=transition_max_inflight,
            transition=self.transition,
        )
        self._warned_empty_candidate_nodes = set()
        self._deferred_eval_tasks = []
        self._deferred_reward_targets = []

    def assign_transition_dispatcher(self, transition_dispatcher_type: str, max_transition_inflight: int, transition: Transition):
        if transition_dispatcher_type == "thread":
            self.transition_dispatcher = ThreadedTransitionDispatcher(transition=transition, max_inflight=max_transition_inflight)
        elif transition_dispatcher_type == "mpi":
            if self.mpi_worker_pool is None:
                raise ValueError("transition_dispatcher_type='mpi' requires inflight_type='mpi'.")
            self.transition_dispatcher = MPITransitionDispatcher(transition=transition, pool=self.mpi_worker_pool)
        else:
            raise ValueError(f"Unknown transition_dispatcher_type: {transition_dispatcher_type}")

    def _resolve_inflight_limits(self, inflight_type: str, max_reward_inflight: int, max_transition_inflight: int, max_mpi_inflight: int) -> tuple[int, int]:
        if inflight_type == "separate":
            if max_mpi_inflight is not None:
                raise ValueError("max_mpi_inflight is only supported when inflight_type='mpi'.")
            if max_reward_inflight is None or max_transition_inflight is None:
                raise ValueError(
                    "inflight_type='separate' requires both max_reward_inflight and max_transition_inflight. "
                    "Set max_reward_inflight for reward jobs and max_transition_inflight for transition jobs."
                )
            if max_reward_inflight <= 0:
                raise ValueError("max_reward_inflight must be >= 1")
            if max_transition_inflight <= 0:
                raise ValueError("max_transition_inflight must be >= 1")
            return max_reward_inflight, max_transition_inflight

        if inflight_type == "mpi":
            if max_mpi_inflight is None:
                raise ValueError("inflight_type='mpi' requires max_mpi_inflight.")
            if max_reward_inflight is not None or max_transition_inflight is not None:
                raise ValueError("max_reward_inflight and max_transition_inflight are not used when inflight_type='mpi'.")
            if max_mpi_inflight <= 0:
                raise ValueError("max_mpi_inflight must be >= 1")
            return max_mpi_inflight, max_mpi_inflight

        raise ValueError("inflight_type must be 'separate' or 'mpi'.")

    def _reward_inflight(self) -> int:
        if self.dispatcher is None:
            return 0
        return self.dispatcher.inflight()

    def _transition_inflight(self) -> int:
        return self.transition_dispatcher.inflight()

    def _reward_can_accept(self) -> bool:
        if self.dispatcher is None:
            return True
        return self.dispatcher.inflight() < self.dispatcher.max_inflight()

    def _transition_can_accept(self) -> bool:
        return self.transition_dispatcher.inflight() < self.transition_dispatcher.max_inflight()

    def _generate_impl(self):
        self._drain_ready_results()
        self._drain_transition_results()
        self._flush_deferred_reward_targets()
        self._flush_deferred_eval_tasks()

        if self._deferred_reward_targets or self._deferred_eval_tasks:
            if self._reward_inflight() > 0 or self._transition_inflight() > 0:
                time.sleep(self.check_interval)
                self._drain_ready_results()
                self._drain_transition_results()
                self._flush_deferred_reward_targets()
                self._flush_deferred_eval_tasks()
            return

        made_progress = False
        if self._reward_can_accept() or self._transition_can_accept():
            made_progress = self._fill_queue()

        if not made_progress and (self._reward_inflight() > 0 or self._transition_inflight() > 0 or self.root.frozen):
            time.sleep(self.check_interval)
            self._drain_ready_results()
            self._drain_transition_results()
            self._flush_deferred_reward_targets()
            self._flush_deferred_eval_tasks()

    def _selection(self) -> Node | None:
        while True:
            node = self.root

            if self.root.frozen:
                return None

            if not self.root.children and (self.root.n > 1 or self.root.is_terminal()):
                self.logger.info("Search tree exhausted.")
                raise SystemExit

            restart = False
            while node.children:
                candidates = self.policy.candidates(node)
                if not candidates:
                    if node is self.root:
                        return None

                    key = node.key()
                    if key not in self._warned_empty_candidate_nodes:
                        self.logger.warning("Node has children but no selectable candidates; freezing it and restarting selection: %s", key)
                        self._warned_empty_candidate_nodes.add(key)
                    node.freeze(recursive=True)
                    restart = True
                    break

                node = self.policy.select_child(node)

            if not restart:
                return node

    def _fill_queue(self) -> bool:
        node = self._selection()
        if node is None:
            return False

        if not node.children and node.n != 0:
            if self.max_tree_depth is not None and node.depth > self.max_tree_depth:
                node.mark_as_terminal(cut=self.cut_terminal, logger=self.logger)
                if self.terminal_reward != "ignore":
                    self._backpropagate(node, self.terminal_reward, False)
                return True
            return self._put_expand_task(node)

        return self._schedule_evaluations_from_node(node)

    def _put_expand_task(self, node: Node) -> bool:
        if not self._transition_can_accept():
            return False

        self._apply_transition_loss(node)
        node.freeze(recursive=True)
        submitted = self.transition_dispatcher.submit(TransitionTask(kind="expand", node=node))
        if not submitted:
            self._revert_transition_loss(node)
            node.unfreeze(recursive=True)
        return submitted

    def _put_reward_task(self, child: Node) -> bool:
        return self._schedule_one(child, self.n_eval_iters, self.n_tries, False)

    def _schedule_evaluations_from_node(self, node: Node) -> bool:
        if node.is_terminal():
            if self.terminal_reward != "ignore":
                self._backpropagate(node, self.terminal_reward, False)
            return True

        if not node.children:
            children = [node]
        elif self.n_eval_width <= 0:
            if not self.policy.candidates(node):
                return False
            children = [self.policy.select_child(node)]
        else:
            children = self.policy.sample_candidates(node, max_size=self.n_eval_width, replace=self.allow_eval_overlaps)

        if children is None or len(children) == 0:
            return False

        self.parent_unfiltered_flag = False
        self.current_parent = node

        made_progress = False
        for child in children:
            submitted = self._put_reward_task(child)
            if not submitted:
                self._defer_eval(child, self.n_eval_iters, self.n_tries, False)
            made_progress = submitted or made_progress
        return made_progress

    def _schedule_one(self, child: Node, iters: int, tries: int, unfiltered_flag: bool) -> bool:
        if child.has_reward():
            return self._handle_reward_target(
                child=child,
                iters=iters,
                tries=tries,
                unfiltered_flag=unfiltered_flag,
                target=child,
                is_direct=True,
            )

        if not self._transition_can_accept():
            return False

        self._apply_transition_loss(child)
        child.freeze(recursive=True)
        task = TransitionTask(
            kind="rollout",
            node=child,
            iters_left=iters,
            tries_left=tries,
            unfiltered_flag=unfiltered_flag,
        )
        submitted = self.transition_dispatcher.submit(task)
        if not submitted:
            self._revert_transition_loss(child)
            child.unfreeze(recursive=True)
        return submitted

    def _handle_reward_target(self, child: Node, iters: int, tries: int, unfiltered_flag: bool, target: Node, is_direct: bool) -> bool:
        if not self._reward_can_accept():
            return False

        pre = self._pre_reward_checks(target)

        if not (type(pre[0]) is bool and pre[0] is True): # no reward calculation
            objective_values, reward = pre
            self.policy.observe(child=child, objective_values=objective_values, reward=reward, is_filtered=(type(objective_values[0])==str))

            if type(objective_values[0]) != str:
                unfiltered_flag = True
                self._backpropagate(child, reward, self.use_dummy_reward)
            else:
                if tries > 1:
                    submitted = self._schedule_one(child, iters, tries-1, unfiltered_flag)
                    if not submitted:
                        self._defer_eval(child, iters, tries-1, unfiltered_flag)
                    return submitted
                elif self.filter_reward[int(objective_values[0])] != "ignore":
                    self._backpropagate(child, self.filter_reward[int(objective_values[0])], False)

            if iters > 1:
                submitted = self._schedule_one(child, iters-1, self.n_tries, unfiltered_flag)
                if not submitted:
                    self._defer_eval(child, iters-1, self.n_tries, unfiltered_flag)
                return submitted
            elif self.cut_failed_child and not unfiltered_flag:
                child.leave(logger=self.logger)
            return True

        key = pre[1]

        if self.reward_dispatcher_type in (None, "disable"):
            return self._run_reward_synchronously(child, iters, tries, unfiltered_flag, target, is_direct, key)

        task = RewardTask(child=child, iters_left=iters, tries_left=tries, unfiltered_flag=unfiltered_flag, target=target, is_direct=is_direct, key=key)
        submitted = self.dispatcher.submit(task)
        if submitted:
            self._apply_virtual_loss(child)
        return submitted

    def _run_reward_synchronously(self, child: Node, iters: int, tries: int, unfiltered_flag: bool, target: Node, is_direct: bool, key: str) -> bool:
        objective_values, reward = self.reward.objective_values_and_reward(target)
        self._post_reward_side_effects(target, key, objective_values, reward)
        task = RewardTask(child=child, iters_left=iters, tries_left=tries, unfiltered_flag=unfiltered_flag, target=target, is_direct=is_direct, key=key)
        self._finish_reward_task(task, objective_values, reward)
        return True

    def _drain_transition_results(self):
        results = self.transition_dispatcher.pop_ready()
        if not results:
            return

        for res in results:
            task = res.task
            node = task.node
            self._revert_transition_loss(node)
            node.unfreeze(recursive=True)

            if res.error is not None:
                raise RuntimeError(f"Transition task failed for node {node.key()}: {res.error}") from res.error

            if task.kind == "expand":
                self._handle_expand_result(node, res.nexts)
            elif task.kind == "rollout":
                submitted = self._handle_reward_target(
                    child=node,
                    iters=task.iters_left,
                    tries=task.tries_left,
                    unfiltered_flag=task.unfiltered_flag,
                    target=res.target,
                    is_direct=False,
                )
                if not submitted:
                    self._defer_reward_target(
                        child=node,
                        iters=task.iters_left,
                        tries=task.tries_left,
                        unfiltered_flag=task.unfiltered_flag,
                        target=res.target,
                        is_direct=False,
                    )
            else:
                raise ValueError(f"Unknown transition task kind: {task.kind}")

    def _drain_ready_results(self):
        if self.reward_dispatcher_type in (None, "disable"):
            return

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
            self._finish_reward_task(task, res.objective_values, res.reward)

    def _finish_reward_task(self, task: RewardTask, objective_values: list, reward: float):
        child = task.child

        if task.is_direct and self.reward_cutoff is not None and reward < self.reward_cutoff and self.reward_cutoff_warmups < self.n_generated_nodes():
            self.reward_cutoff_count += 1
            child.leave(logger=self.logger)

        self.policy.observe(child=child, objective_values=objective_values, reward=reward, is_filtered=False)

        task.unfiltered_flag = True
        self._backpropagate(child, reward, self.use_dummy_reward)

        if task.iters_left > 1:
            submitted = self._schedule_one(child, task.iters_left - 1, self.n_tries, task.unfiltered_flag)
            if not submitted:
                self._defer_eval(child, task.iters_left - 1, self.n_tries, task.unfiltered_flag)

    def _handle_expand_result(self, node: Node, nexts: list[Node]):
        if self.discard_unneeded_states:
            node.discard_unneeded_states()

        expanded = False
        for n in nexts:
            if self.avoid_duplicates:
                key = n.key()
                if key in self.node_keys:
                    continue
                self.node_keys.add(key)
            node.add_child(n, override_parent=True)
            expanded = True

        if not expanded:
            node.mark_as_terminal(cut=self.cut_terminal, logger=self.logger)

        self._schedule_evaluations_from_node(node)

    def _apply_transition_loss(self, node: Node):
        cur = node
        while cur is not None:
            cur.transition_loss_count += 1
            cur.n += 1
            cur.sum_r += self.transition_loss
            cur = cur.parent

    def _revert_transition_loss(self, node: Node):
        cur = node
        while cur is not None:
            if cur.transition_loss_count <= 0:
                cur = cur.parent
                continue

            cur.transition_loss_count -= 1
            cur.n -= 1
            cur.sum_r -= self.transition_loss
            cur = cur.parent

    def _defer_eval(self, child: Node, iters: int, tries: int, unfiltered_flag: bool):
        self._deferred_eval_tasks.append((child, iters, tries, unfiltered_flag))

    def _flush_deferred_eval_tasks(self) -> bool:
        if not self._deferred_eval_tasks:
            return False

        remaining = []
        made_progress = False
        for child, iters, tries, unfiltered_flag in self._deferred_eval_tasks:
            submitted = self._schedule_one(child, iters, tries, unfiltered_flag)
            if submitted:
                made_progress = True
            else:
                remaining.append((child, iters, tries, unfiltered_flag))
        self._deferred_eval_tasks = remaining
        return made_progress

    def _defer_reward_target(self, child: Node, iters: int, tries: int, unfiltered_flag: bool, target: Node, is_direct: bool):
        self._deferred_reward_targets.append((child, iters, tries, unfiltered_flag, target, is_direct))

    def _flush_deferred_reward_targets(self) -> bool:
        if not self._deferred_reward_targets:
            return False

        remaining = []
        made_progress = False
        for child, iters, tries, unfiltered_flag, target, is_direct in self._deferred_reward_targets:
            submitted = self._handle_reward_target(child, iters, tries, unfiltered_flag, target, is_direct)
            if submitted:
                made_progress = True
            else:
                remaining.append((child, iters, tries, unfiltered_flag, target, is_direct))
        self._deferred_reward_targets = remaining
        return made_progress
