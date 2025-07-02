from .base import Reward, MolReward
from .log_p_reward import LogPReward

# lazy import
def __getattr__(name):
    if name == "DScoreReward":
        from .d_score_reward import DScoreReward
        return DScoreReward
    if name == "JScoreReward":
        from .j_score_reward import JScoreReward
        return JScoreReward
    raise AttributeError(f"module {__name__} has no attribute {name}")