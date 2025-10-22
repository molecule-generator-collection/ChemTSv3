from .base import Transition, AutoRegressiveTransition, TemplateTransition, BlackBoxTransition, LLMTransition
from .gpt2 import GPT2Transition
from .gbga import GBGATransition
from .rnn import RNNLanguageModel, RNNTransition
from .smirks import SMIRKSTransition

# lazy import
def __getattr__(name):
    if name == "BioT5Transition":
        from .biot5 import BioT5Transition
        return BioT5Transition
    if name == "ChatGPTTransition":
        from .chat_gpt import ChatGPTTransition
        return ChatGPTTransition
    if name == "ChatGPTTransitionWithMemory":
        from .chat_gpt_with_memory import ChatGPTTransitionWithMemory
        return ChatGPTTransitionWithMemory
    if name == "ProtGPT2Transition":
        from .prot_gpt2 import ProtGPT2Transition
        return ProtGPT2Transition
    if name == "RNNBasedMutation":
        from .rnn_based_mutation import RNNBasedMutation
        return RNNBasedMutation
    raise AttributeError(f"module {__name__} has no attribute {name}")