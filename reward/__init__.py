from .base import Reward, MolReward, SMILESReward
from .log_p_reward import LogPReward

# lazy import
def __getattr__(name):
    if name == "DScoreReward":
        from .d_score_reward import DScoreReward
        return DScoreReward
    if name == "DyRAMOReward":
        from .dyramo_reward import DyRAMOReward
        return DyRAMOReward
    if name == "EGFRReward":
        from .egfr_reward import EGFRReward
        return EGFRReward
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