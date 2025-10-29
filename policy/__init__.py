from .base import Policy, ScoreBasedPolicy
from .uct import UCT
from .puct import PUCT

def __getattr__(name):
    if name == "PUCTWithPredictor":
        from .puct_with_predictor import PUCTWithPredictor
        return PUCTWithPredictor
    raise AttributeError(f"module {__name__} has no attribute {name}")