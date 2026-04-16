from .base import Generator
from .heapq_generator import HeapQueueGenerator
from .mcts import MCTS
from .random_generator import RandomGenerator

# lazy import
def __getattr__(name):
    if name == "AsyncParallelMCTS":
        from .async_parallel_mcts import AsyncParallelMCTS
        return AsyncParallelMCTS
    if name == "MPIRewardDispatcher":
        from .async_parallel_mcts import MPIRewardDispatcher
        return MPIRewardDispatcher
    if name == "AsyncParallelMCTS":
        from .async_parallel_mcts import worker_loop
        return worker_loop
    raise AttributeError(f"module {__name__} has no attribute {name}")