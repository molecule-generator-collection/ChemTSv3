from bisect import bisect_right
import math
from typing import Callable
import numpy as np

#used for expansion_threshold
def select_indices_by_threshold(probs: list[float], threshold: float) -> list[int]:
    probs = np.array(probs)
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, threshold)
    return sorted_indices[:cutoff + 1].tolist()

def moving_average(values: list[float], window: float=0.05) -> np.ndarray:
    if window < 1:
        window = math.floor(len(values) * window)
    head = [np.mean(values[:i+1]) for i in range(window - 1)]
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
