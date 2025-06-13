import logging
import os
from typing import Any, Self
import torch
import torch.nn.functional as F
from language import Language
from node import SentenceNode
from transition import LanguageModel

class RNNTransition(LanguageModel):
    def __init__(self, lang: Language, model_dir: str=None, name: str=None, device: str=None, logger: logging.Logger=None):
        super().__init__(lang=lang, name=name, logger=logger)
    
    #implement
    def _transitions_with_probs_impl(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
        pass
    
    def rollout(self, initial_node: SentenceNode, top_p=0.995) -> SentenceNode:
        pass