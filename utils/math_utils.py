from bisect import bisect_right
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

def make_curve_from_points(points: list[tuple[float, float]]) -> Callable[[float], float]:
    if not points:
        raise ValueError("Points must not be empty")

    points.sort()
    xs, ys = zip(*points)

    def curve(x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]

        i = bisect_right(xs, x)
        x0, y0 = xs[i - 1], ys[i - 1]
        x1, y1 = xs[i], ys[i]

        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    return curve

def minmax(x, min, max):
    return (x - min) / (max - min)

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