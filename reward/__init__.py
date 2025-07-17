from .base import Reward, MolReward
from .log_p_reward import LogPReward

# lazy import
def __getattr__(name):
    if name == "DScoreReward":
        from .d_score_reward import DScoreReward
        return DScoreReward
    if name == "GuacaMolReward":
        from .guacamol_reward import GuacaMolReward
        return GuacaMolReward
    if name == "JScoreReward":
        from .j_score_reward import JScoreReward
        return JScoreReward
    if name == "TDCReward":
        from .tdc_reward import TDCReward
        return TDCReward
    raise AttributeError(f"module {__name__} has no attribute {name}")