from .base import Reward, SingleReward, MolReward, SingleMolReward, SMILESReward
from .log_p_reward import LogPReward
from .similarity_reward import SimilarityReward

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
    if name == "JScoreReward":
        from .j_score_reward import JScoreReward
        return JScoreReward
    if name == "GFPReward":
        from .gfp_reward import GFPReward
        return GFPReward
    if name == "TDCReward":
        from .tdc_reward import TDCReward
        return TDCReward
    raise AttributeError(f"module {__name__} has no attribute {name}")