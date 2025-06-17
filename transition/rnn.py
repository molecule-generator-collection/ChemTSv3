import json
import logging
import os
from typing import Any, Self
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from language import Language
from node import SentenceNode
from transition import LanguageModel

class RNNLanguageModel(nn.Module):
    def __init__(self, pad_id: int, vocab_size: int, embed_size: int=None, hidden_size: int=256, num_layers: int=2, rnn_type: str="GRU", dropout: float=0.3, use_input_dropout=True):
        super().__init__()
        self.vocab_size = vocab_size
        embed_size = embed_size or vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.use_input_dropout = use_input_dropout
        if use_input_dropout:
            self.dropout_in = nn.Dropout(dropout)
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[rnn_type]
        self.rnn = rnn_cls(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def _init_states(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.rnn_type == "LSTM":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None):
        """
        x: [batch, seq_len] (LongTensor)
        returns logits: [batch, seq_len, vocab_size], next_hidden
        """
        lengths = (x != self.embed.padding_idx).sum(dim=1)
        if hidden is None:
            hidden = self._init_states(x.size(0), x.device)
        if self.use_input_dropout:
            emb = self.dropout_in(self.embed(x))
        else:
            emb = self.embed(x)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed, hidden)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        return logits, hidden

    @torch.inference_mode()
    def generate(self,input_ids: torch.Tensor, max_length: int, eos_token_id: int, pad_token_id: int, top_p: float=1.0) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        hidden = None

        for _ in range(max_length - input_ids.size(1)):
            logits, hidden = self(generated[:, -1:], hidden)
            next_logits = logits[:, -1, :]  # [1, vocab]
            probs = F.softmax(next_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative <= top_p
                mask[..., 0] = True
                filtered_indices = indices[mask]
                filtered_probs = sorted_probs[mask]
                filtered_probs = filtered_probs / filtered_probs.sum()
                next_id = filtered_indices[torch.multinomial(filtered_probs, num_samples=1)].unsqueeze(0)
            else:
                next_id = torch.multinomial(probs, num_samples=1)               

            generated = torch.cat([generated, next_id], dim=1)

            if next_id.item() == eos_token_id:
                break

        return generated
    
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))
        cfg = {
            "vocab_size": self.vocab_size,
            "embed_size": self.embed.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

class RNNTransition(LanguageModel):
    def __init__(self, lang: Language, model: RNNLanguageModel=None, model_dir: str=None, max_length=None, top_p=1.0, name: str=None, device: str=None, logger: logging.Logger=None):
        if (model is not None) and (model_dir is not None):
            raise ValueError("specify one (or none) of model or model_dir, not both.")
        
        super().__init__(lang=lang, name=name, logger=logger)
        self.logger.info("Is CUDA available: " + str(torch.cuda.is_available()))

        device = self.lang.device
        if model is not None:
            self.model = model.to(device)
        elif model_dir is not None:
            self.load(model_dir, device=device)
        
        self._max_length = max_length or 10**18
        self.top_p = top_p        
        
    def load(self, model_dir: str, device: str=None) -> Self:
        """
        model_dir:
            ├─ model.pt (state_dict)
            └─ config.json (RNN hyperparams)
        """
        if device is None:
            device = self.lang.device
        device = torch.device(device or "cpu")
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        self.model = RNNLanguageModel(**cfg).to(device)
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
        self.model.load_state_dict(state)
        self.name = os.path.basename(os.path.normpath(model_dir))
        return self
    
    # override
    def max_length(self):
        return self._max_length

    #implement
    def _transitions_with_probs_impl(self, node: SentenceNode) -> list[tuple[Any, SentenceNode, float]]:
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(node.id_tensor.to(self.lang.device))
            next_logits = logits[0, -1, :]  # [vocab]
            probs = F.softmax(next_logits, dim=-1).tolist()

        children = []
        for tok_id, prob in enumerate(probs):
            next_tensor = torch.cat([node.id_tensor, self.lang.list2tensor([tok_id]).to(self.lang.device)], dim=1)
            child = node.__class__(id_tensor=next_tensor, lang=node.lang, parent=node, last_prob=prob)
            children.append((tok_id, child, prob))
        return children
    
    def rollout(self, initial_node: SentenceNode) -> SentenceNode:
        with torch.no_grad():
            generated_tensor = self.model.generate(
                input_ids=initial_node.id_tensor.to(self.lang.device),
                max_length=self.max_length(),
                eos_token_id=self.lang.eos_id(),
                pad_token_id=self.lang.pad_id(),
                top_p=self.top_p,
            )
        return initial_node.__class__(id_tensor=generated_tensor.to(initial_node.id_tensor.device), lang=self.lang)