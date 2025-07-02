from .base import Reward, MolReward
from .j_score_reward import JScoreReward
from .log_p_reward import LogPReward

# lazy import
def __getattr__(name):
    if name == "DScoreReward":
        from .d_score_reward import DScoreReward
        return DScoreReward
    raise AttributeError(f"module {__name__} has no attribute {name}")