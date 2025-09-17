from .base import Transition, LanguageModel, BlackBoxTransition, TemplateTransition
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
        from .biot5 import BioT5PlusTransition
        return BioT5PlusTransition
    if name == "ChatGPTTransition":
        from .chat_gpt import ChatGPTTransition
        return ChatGPTTransition
    if name == "LongChatGPTTransition":
        from .chat_gpt import LongChatGPTTransition
        return LongChatGPTTransition
    if name == "ProtGPT2Transition":
        from .prot_gpt2 import ProtGPT2Transition
        return ProtGPT2Transition
    raise AttributeError(f"module {__name__} has no attribute {name}")