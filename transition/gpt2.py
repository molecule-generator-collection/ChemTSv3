import logging
import os
from typing import Any, Self
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from language import Language
from node import SentenceNode
from transition import LanguageModel
from utils import apply_top_p

class GPT2Transition(LanguageModel):
    def __init__(self, lang: Language, model=None, model_dir: str=None, device: str=None, logger: logging.Logger=None, temperature: float=1.0, top_p: float=0.995, top_k: int=0, repetition_penalty: float=1.0):
        # TODO: either remove repetition_penalty / top_k or implement to transition_with_probs
        # TODO: might move shared codes with RNN
        if (model is not None) and (model_dir is not None):
            raise ValueError("specify either 'model' or 'model_dir', not both.")

        if model is not None:
            self.model = model
        elif model_dir is not None:
            self.load(model_dir, device=device)
            
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        super().__init__(lang=lang, logger=logger)
        if device != "cpu":
            self.logger.info("Is CUDA available: " + str(torch.cuda.is_available()))

    def load(self, model_dir: str, device: str=None) -> Self:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(torch.device(self.device))
        self.name = os.path.basename(os.path.normpath(model_dir))
        return self

    # override
    def max_length(self):
        return self.model.config.n_positions

    # implement
    def transitions_with_probs(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
        if node.id_tensor[0][-1] == self.lang.eos_id():
            return []

        with torch.no_grad():
            outputs = self.model(node.id_tensor)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            next_logits = logits[:, -1, :]
            next_logits = next_logits / self.temperature
            probs = F.softmax(next_logits, dim=-1)
        if self.top_p < 1.0:
            probs = apply_top_p(probs, top_p=self.top_p)
        probs = probs.tolist()[0]
        
        children = []
        for tok_id, prob in enumerate(probs):
            next_tensor = torch.cat([node.id_tensor, self.lang.list2tensor([tok_id]).to(self.device)], dim=1)
            if prob != 0:
                child = node.__class__(id_tensor=next_tensor, lang=node.lang, parent=node, last_prob=prob, last_action=tok_id)
                children.append((tok_id, child, prob))
        return children

    # override
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
        return initial_node.__class__(id_tensor=result_tensor, lang=self.lang)
