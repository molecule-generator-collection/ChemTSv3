from .base import Language, DynamicLanguage, MolLanguage, DynamicMolLanguage
from .helm import HELM
from .smiles import SMILES

# lazy import
def __getattr__(name):
    if name == "DScoreReward":
        from .selfies import SELFIES
        return SELFIES
    raise AttributeError(f"module {__name__} has no attribute {name}")