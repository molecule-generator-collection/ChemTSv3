import os
from typing import Any
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from language import Language
from transition import LanguageModel
from node import SentenceNode

class GPT2Transition(LanguageModel):
  def __init__(self, lang: Language, model=None, model_dir=None, name=None):
    assert (model is not None) or (model_dir is not None), \
            "specify model or model_dir."
    assert (model is None) or (model_dir is None), \
            "specify one of model or model_dir, not both."

    if model is not None:
      self.model = model
    else:
      print("Is CUDA available: " + str(torch.cuda.is_available()))
      self.model = GPT2LMHeadModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
      if name is None:
        name = os.path.basename(os.path.normpath(model_dir))

    super().__init__(lang=lang, name=name)

  #override
  def max_length(self):
    return self.model.config.n_positions

  #override
  def transitions_with_probs(self, node: SentenceNode) -> list[SentenceNode]:
    nodes = []

    with torch.no_grad():
      outputs = self.model(node.id_tensor)
      logits = outputs.logits  #shape: [batch_size, seq_len, vocab_size]
      next_token_logits = logits[0, -1, :]

    probs = F.softmax(next_token_logits, dim=-1).tolist()

    for i in range(len(probs)):
      nodes.append(node.__class__(id_tensor=torch.cat([node.id_tensor, Language.list2tensor([i])], dim=1), lang=node.lang, parent=node, last_prob=probs[i]))

    return [(i, nodes[i], probs[i]) for i in range(len(probs))]

  #override
  def rollout(self, initial_node: SentenceNode, rollout_threshold=0.995) -> SentenceNode:
    #rollout_threshold: [0-1], ignore children with low transition probabilities in rollout based on this value
    with torch.no_grad():
        result_tensor = self.model.generate(
            initial_node.id_tensor,
            max_length=self.max_length(),
            do_sample=True,       #sampling
            #top_k=50,             #top-k sampling
            top_p=rollout_threshold,  #nucleus sampling
            eos_token_id=self.lang.eos_id(),
            pad_token_id=self.lang.pad_id(),
            num_return_sequences=1
        )
    result = initial_node.__class__(id_tensor=result_tensor, lang=self.lang)
    return result
