from .base import Transition, LanguageModel, BlackBoxTransition
from .gpt2 import GPT2Transition
from .jensen import JensenTransition
from .rnn import RNNLanguageModel, RNNTransition
from .smirks import SMIRKSTransition

# lazy import
def __getattr__(name):
    if name == "BioT5Transition":
        from .biot5 import BioT5Transition
        return BioT5Transition
    if name == "BioT5PlusTransition":
        from .biot5_plus import BioT5PlusTransition
        return BioT5PlusTransition
    raise AttributeError(f"module {__name__} has no attribute {name}")