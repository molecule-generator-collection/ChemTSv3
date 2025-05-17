import numpy as np

#used for expansion_threshold
def select_indices_by_threshold(probs: list[float], threshold: float) -> list[int]:
    probs = np.array(probs)
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, threshold)
    return sorted_indices[:cutoff + 1].tolist()