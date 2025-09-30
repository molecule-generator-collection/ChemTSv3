from bisect import bisect_right
import logging
import math
import random
import time
import numpy as np
import pandas as pd
import torch

def set_seed(seed: int=None, logger: logging.Logger=None):
    if seed is None:    
        seed = int(time.time()*1000) % (2**32)
        
    if logger is None:
        print("seed: " + str(seed))
    else:
        logger.info("seed: " + str(seed))
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def append_pareto_optimality_to_df(df: pd.DataFrame, objectives: list[str], maximize: list[bool]=None, colname: str="is_pareto_optimal") -> pd.DataFrame:
    """
    Add a boolean column to df indicating whether each row is Pareto optimal. For more than 2 objectives, pymoo is required.

    Args:
        df: Input dataframe.
        objectives: List of objective column names.
        maximize: Same length as objectives. If True, higher values are better. If None, defaults to all True.
        colname: Name of the output column.

    Returns:
        pd.DataFrame: Dataframe with added column for Pareto optimality.
    """
    if maximize is None:
        maximize = [True] * len(objectives)
    if len(maximize) != len(objectives):
        raise ValueError("`maximize` must have the same length as `objectives`.")

    F = df[objectives].to_numpy(dtype=np.float32)

    # Switch to maximize
    for k, is_max in enumerate(maximize):
        if not is_max:
            F[:, k] = -F[:, k]

    try:
        # If pymoo is available
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        nds = NonDominatedSorting(method="efficient_non_dominated_sort")
        first_front = nds.do(F, only_non_dominated_front=True) # Get non-dominated indices.
        mask = np.zeros(len(F), dtype=bool)
        mask[np.asarray(first_front, dtype=int)] = True

    except ImportError as e:
        # If pymoo is not available, fall back to a 2D O(N log N) skyline for m==2. For m >= 3 without pymoo, raise an error.
        if F.shape[1] == 2:
            order = np.argsort(-F[:, 0], kind="mergesort")
            best2 = -np.inf
            mask = np.zeros(F.shape[0], dtype=bool)
            for idx in order:
                v2 = F[idx, 1]
                if v2 > best2:
                    mask[idx] = True
                    best2 = v2
        else:
            raise RuntimeError("Please install pymoo.") from e

    df[colname] = mask
    return df

def pareto_optimal_df(df: pd.DataFrame, objectives: list[str], maximize: list[bool] = None) -> pd.DataFrame:
    """
    Args:
        df: Input dataframe
        objectives: List of objective column names
        maximize: Same length as objectives. If True, higher values are better. If None, defaults to all True.

    Returns:
        pd.DataFrame: Dataframe with Pareto optimal entries
    """
    df = append_pareto_optimality_to_df(df, objectives=objectives, maximize=maximize, colname="is_pareto")
    result = df.loc[df["is_pareto"]]
    df.drop(columns=["is_pareto"], inplace=True)
    result = result.drop(columns=["is_pareto"])

    return result

def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= top_p
    mask[..., 0] = True  # at least one

    filtered_indices = sorted_indices[mask]
    filtered_probs = sorted_probs[mask]

    original_order = torch.argsort(filtered_indices)
    filtered_indices = filtered_indices[original_order]
    filtered_probs = filtered_probs[original_order]

    filtered_probs = filtered_probs / filtered_probs.sum()

    new_probs = torch.zeros_like(probs)
    new_probs[0, filtered_indices] = filtered_probs

    return new_probs

def apply_sharpness(probs: torch.Tensor, sharpness: float) -> torch.Tensor:
    powered = torch.pow(probs, sharpness)
    normalized = powered / powered.sum(dim=-1, keepdim=True)
    return normalized

def moving_average(values: list[float], window: float=0.05, top_p: float=None) -> np.ndarray:
    if window < 1:
        window = max(1, math.floor(len(values) * window))
    window = min(window, len(values))
    
    head = [np.nan] * (window - 1)
    if top_p is not None and 0 < top_p < 1:
        tail = []
        for i in range(len(values) - window + 1):
            window_values = values[i:i+window]
            sorted_vals = sorted(window_values, reverse=True)
            top_k = max(1, math.floor(len(sorted_vals) * top_p))
            tail.append(np.mean(sorted_vals[:top_k]))
    else:
        tail = np.convolve(values, np.ones(window)/window, mode='valid')
    return np.array(head + list(tail))

def max_gauss(x, a=1, mu=8, sigma=2):
    if x > mu:
        return 1
    else:
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def min_gauss(x, a=1, mu=2, sigma=2):
    if x < mu:
        return 1
    else:
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def rectangular(x, min, max):
    if min <= x <= max:
        return 1
    else:
        return 0
    
class PointCurve():
    def __init__(self, points: list[tuple[float, float]]):
        if not points:
            raise ValueError("Points must not be empty")

        points.sort()
        self.xs, self.ys = zip(*points)

    def curve(self, x: float) -> float:
        if x <= self.xs[0]:
            return self.ys[0]
        if x >= self.xs[-1]:
            return self.ys[-1]

        i = bisect_right(self.xs, x)
        x0, y0 = self.xs[i - 1], self.ys[i - 1]
        x1, y1 = self.xs[i], self.ys[i]

        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
