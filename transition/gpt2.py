import logging
import os
from typing import Any, Self
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GenerationConfig
from language import Language
from node import SentenceNode
from transition import LanguageModel

class GPT2Transition(LanguageModel):
    def __init__(self, lang: Language, model=None, model_dir: str=None, name=None, logger: logging.Logger=None, temperature: float=1.0, top_p: float=0.995, top_k: int=0, repetition_penalty: float=1.0):
        assert (model is None) or (model_dir is None), \
            "specify one (or none) of model or model_dir, not both."

        if model is not None:
            self.model = model
        elif model_dir is not None:
            self.load(model_dir, device=lang.device)
            
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        super().__init__(lang=lang, name=name, logger=logger)
        self.logger.info("Is CUDA available: " + str(torch.cuda.is_available()))
    
    def load(self, model_dir: str, device: str=None) -> Self:
        if device is None:
            device = self.lang.device
        self.model = GPT2LMHeadModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(torch.device(device))
        self.name = os.path.basename(os.path.normpath(model_dir))
        return self

    # override
    def max_length(self):
        return self.model.config.n_positions

    # implement
    def _transitions_with_probs_impl(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
        nodes = []

        with torch.no_grad():
            outputs = self.model(node.id_tensor)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]

        probs = F.softmax(next_token_logits, dim=-1).tolist()

        for i in range(len(probs)):
            nodes.append(node.__class__(id_tensor=torch.cat([node.id_tensor, self.lang.list2tensor([i])], dim=1), lang=node.lang, parent=node, last_prob=probs[i]))

        return [(i, nodes[i], probs[i]) for i in range(len(probs))]

    # implement
    def rollout(self, initial_node: SentenceNode) -> SentenceNode:
        """
        Args:
            top_k: inactive if set to 0 / torch's default value is 50
            top_p: [0-1], ignore children with low transition probabilities in rollout based on this value
            repetition_penalty: inactive if set to 1.0
        """
        with torch.no_grad():
            result_tensor = self.model.generate(
                initial_node.id_tensor,
                max_length=self.max_length(),
                do_sample=True, # sampling
                temperature=self.temperature, 
                top_k=self.top_k,
                top_p=self.top_p, # nucleus sampling
                repetition_penalty=self.repetition_penalty, 
                eos_token_id=self.lang.eos_id(),
                pad_token_id=self.lang.pad_id(),
                num_return_sequences=1
            )
        result = initial_node.__class__(id_tensor=result_tensor, lang=self.lang)
        return result
