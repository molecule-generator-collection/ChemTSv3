from bisect import bisect_right
import math
from typing import Callable
import numpy as np
import torch

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

def apply_power(probs: torch.Tensor, power: float) -> torch.Tensor:
    powered = torch.pow(probs, power)
    normalized = powered / powered.sum(dim=-1, keepdim=True)
    return normalized

def moving_average(values: list[float], window: float=0.05) -> np.ndarray:
    if window < 1:
        window = max(1, math.floor(len(values) * window))
    head = [np.nan] * (window - 1)
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
