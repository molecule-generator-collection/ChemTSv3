import json
import logging
import os
from typing import Any, Self
import torch
import torch.nn as nn
import torch.nn.functional as F
from language import Language, DynamicLanguage
from node import SentenceNode
from transition import LanguageModel
from utils import apply_top_p, apply_sharpness

class RNNLanguageModel(nn.Module):
    def __init__(self, pad_id: int, vocab_size: int, embed_size: int=None, hidden_size: int=256, num_layers: int=2, rnn_type: str="GRU", dropout: float=0.3, use_input_dropout=True):
        super().__init__()
        self.vocab_size = vocab_size
        embed_size = embed_size or vocab_size
        self.pad_id = pad_id
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
        Args:
            x: [batch, seq_len] (LongTensor)  
        Returns:
            logits: [batch, seq_len, vocab_size]
            next_hidden
        """
        if hidden is None:
            hidden = self._init_states(x.size(0), x.device)
        if self.use_input_dropout:
            emb = self.dropout_in(self.embed(x))
        else:
            emb = self.embed(x)
        out, next_hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, next_hidden

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_length: int, eos_token_id: int, top_p: float=1.0, temperature: float=1.0, sharpness: float=1.0) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        logits, hidden = self.forward(generated)
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        if top_p < 1.0:
            probs = apply_top_p(probs, top_p)
        if sharpness != 1.0:
            probs = apply_sharpness(probs, sharpness)

        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)

        while (generated.size(1) < max_length and (next_id != eos_token_id).all()):
            logits, hidden = self.forward(next_id, hidden)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            if top_p < 1.0:
                probs = apply_top_p(probs, top_p)
            if sharpness != 1.0:
                probs = apply_sharpness(probs, sharpness)

            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

        return generated
    
    def count_all_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))
        cfg = {
            "vocab_size": self.vocab_size,
            "embed_size": self.embed.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "pad_id": self.pad_id,
            "use_input_dropout": self.use_input_dropout
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

class RNNTransition(LanguageModel):
    def __init__(self, lang: Language, model: RNNLanguageModel=None, model_dir: str=None, device: str=None, max_length=None, top_p=1.0, temperature=1.0, sharpness=1.0, logger: logging.Logger=None):
        if (model is not None) and (model_dir is not None):
            raise ValueError("specify one (or none) of 'model' or 'model_dir', not both.")
        
        super().__init__(lang=lang, logger=logger)
        if device != "cpu":
            self.logger.info("Is CUDA available: " + str(torch.cuda.is_available()))

        if model is not None:
            self.model = model
        elif model_dir is not None:
            self.load(model_dir, device=device)
        
        self._max_length = max_length or 10**18
        self.top_p = top_p
        self.temperature = temperature
        self.sharpness = sharpness
        
    def load(self, model_dir: str, device: str=None) -> Self:
        """
        model_dir:
            ├─ model.pt (state_dict)
            └─ config.json (RNN hyperparams)
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        self.model = RNNLanguageModel(**cfg).to(self.device)
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        self.name = os.path.basename(os.path.normpath(model_dir))
        return self
    
    # override
    def max_length(self):
        return self._max_length

    #implement
    def next_nodes(self, node: SentenceNode) -> list[SentenceNode]:
        if node.id_tensor[0][-1] == self.lang.eos_id():
            return []
        
        self.model.eval()
        with torch.no_grad(): 
            logits, _ = self.model(node.id_tensor.to(self.device))
            next_logits = logits[:, -1, :]
            next_logits = next_logits / self.temperature
            probs = F.softmax(next_logits, dim=-1)
        if self.top_p < 1.0:
            probs = apply_top_p(probs, top_p=self.top_p)
        if self.sharpness != 1.0:
            probs = apply_sharpness(probs, self.sharpness)
        probs = probs.tolist()[0]
        
        children = []
        for tok_id, prob in enumerate(probs):
            next_tensor = torch.cat([node.id_tensor, self.lang.list2tensor([tok_id]).to(self.device)], dim=1)
            if prob != 0:
                child = node.__class__(id_tensor=next_tensor, lang=node.lang, parent=node, last_prob=prob, last_action=tok_id)
                children.append(child)
        return children
    
    # override
    def rollout(self, initial_node: SentenceNode) -> SentenceNode:
        with torch.no_grad():
            generated_tensor = self.model.generate(
                input_ids=initial_node.id_tensor, # .to(self.device)
                max_length=self.max_length(),
                eos_token_id=self.lang.eos_id(),
                top_p=self.top_p,
                temperature=self.temperature,
                sharpness=self.sharpness
            )
        return initial_node.__class__(id_tensor=generated_tensor, lang=self.lang) # .to(initial_node.id_tensor.device)

    @staticmethod
    def train_rnn_with_dynamic_language(lang: DynamicLanguage, dataset_path: str, test_dataset_path: str=None, test_size: float=0.1, batch_size=64, lr=1e-3, num_epochs=10, rnn_type="GRU", embed_size=None, hidden_size=256, num_layers=2, dropout=0.3) -> tuple[Self, dict]:
        """
        Returns:
            Self: last model
            dict: best state dict
        """
        from datasets import load_dataset
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        device="cuda:0" if torch.cuda.is_available() else "cpu"
        print("Is CUDA available: " + str(torch.cuda.is_available()))
        
        # make dataset and build vocabs
        if test_dataset_path is None:
            ds = load_dataset("text", data_files={"train": dataset_path})
            ds = ds["train"].train_test_split(test_size=test_size)
        else:
            ds = load_dataset("text", data_files={"train": dataset_path, "test": test_dataset_path})
        lang.build_vocab(ds)

        ds_tokenized = ds.map(lambda x: {"ids": lang.sentence2ids(x["text"])}, remove_columns=["text"])
        train_dataset = ds_tokenized["train"]
        test_dataset  = ds_tokenized["test"]
        pad_id = lang.pad_id()
        
        def collate(batch):
            seqs = [torch.tensor(ex["ids"]) for ex in batch]
            maxlen = max(len(s) for s in seqs)
            padded = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                padded[i, :len(s)] = s
            return padded[:, :-1], padded[:, 1:] # input, target
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        
        model = RNNLanguageModel(pad_id=lang.pad_id(), vocab_size=len(lang.vocab()), embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, rnn_type=rnn_type, dropout=dropout).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)


        best_val_loss = float('inf')
        best_state_dict = None
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(test_loader)

            print(f"[{epoch}] train_loss: {avg_train_loss:.4f}  val_loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = model.state_dict()
            
        return model, best_state_dict
    
    @staticmethod
    def train_rnn_from_conf(conf: dict, repo_root: str="../") -> tuple[Self, dict, Language]:
        """
        Train RNN from conf. Currently only supports DynamicLanguage.
        
        Returns:
            Self: last model
            dict: best state dict
            Language: language
        """
        from utils import class_from_package
        
        # set language from conf
        lang_class = class_from_package("language", conf.pop("lang_class"))
        lang = lang_class(**conf.pop("lang_args", {}))

        # set path from conf
        conf["dataset_path"]= os.path.join(repo_root, conf["dataset_path"])
        if "test_dataset_path" in conf:
            conf["test_dataset_path"] = os.path.join(repo_root, conf["test_dataset_path"])
            
        model, best_state_dict = RNNTransition.train_rnn_with_dynamic_language(lang=lang, **conf)
        return model, best_state_dict, lang