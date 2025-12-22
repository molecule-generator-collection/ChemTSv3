from .base import Reward, SingleReward, MolReward, SingleMolReward, SMILESReward, BatchReward
from .log_p_reward import LogPReward
from .similarity_reward import SimilarityReward

# lazy import
def __getattr__(name):
    if name == "JScoreReward":
        from .j_score_reward import JScoreReward
        return JScoreReward
    raise AttributeError(f"module {__name__} has no attribute {name}")