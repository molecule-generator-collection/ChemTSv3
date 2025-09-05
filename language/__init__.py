from .base import Language, DynamicLanguage, MolLanguage, DynamicMolLanguage
from .fasta import FASTA
from .helm import HELM
from .smiles import SMILES

# lazy import
def __getattr__(name):
    if name == "SELFIES":
        from .selfies import SELFIES
        return SELFIES
    if name == "TokenizerLanguage":
        from .tokenizer import TokenizerLanguage
        return TokenizerLanguage
    raise AttributeError(f"module {__name__} has no attribute {name}")