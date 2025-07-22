from abc import ABC, abstractmethod
from collections import Counter
import pickle
from typing import Self
from rdkit.Chem import Mol
import torch

# vocabulary can be dynamic for better compatibility, thus most methods are not static
class Language(ABC):
    _bos_token = "<BOS>"
    _eos_token = "<EOS>"
    _pad_token = "<PAD>"
    _unk_token = "<UNKNOWN>"

    @abstractmethod
    def sentence2ids(self, sentence: str, include_eos: bool=True) -> list[int]:
        """Convert sentence to token ids"""
        pass
    
    @abstractmethod
    def token2id(self, token: str) -> int:
        pass

    @abstractmethod
    def id2token(tokenid: int) -> str:
        pass

    @abstractmethod
    def vocab(self) -> list[str]:
        """List of all possible tokens, can be dynamic (thus not a static method)"""
        pass

    @abstractmethod
    def ids2sentence(self, ids: list[int]) -> str:
        """Revert the token id sequence to sentence"""
        pass
    
    def bos_token(self) -> str:
        return self.__class__._bos_token
    
    def eos_token(self) -> str:
        return self.__class__._eos_token
    
    def pad_token(self) -> str:
        return self.__class__._pad_token
    
    def unk_token(self) -> str:
        return self.__class__._unk_token
    
    def eos_id(self) -> int:
        return self.token2id(self.__class__._eos_token)
    
    def bos_id(self) -> int:
        return self.token2id(self.__class__._bos_token)
    
    def pad_id(self) -> int:
        return self.token2id(self.__class__._pad_token)
    
    def unk_id(self) -> int:
        return self.token2id(self.__class__._unk_token)
    
    def list2tensor(self, ids: list[int], device: str=None) -> torch.Tensor:
        return torch.tensor([ids], device=torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu")))
    
    @staticmethod
    def tensor2list(t: torch.Tensor) -> list[int]:
        return t[0].tolist()
    
    def tensor2sentence(self, tensor: torch.Tensor) -> str:
        l = self.tensor2list(tensor)
        return self.ids2sentence(l)
    
    def sentence2tensor(self, sentence: str, include_eos: bool=True, device: str=None) -> torch.Tensor:
        l = self.sentence2ids(sentence, include_eos=include_eos)
        return self.list2tensor(l, device=device)
    
    def bos_tensor(self, device: str=None):
        return self.list2tensor([self.bos_id()], device=device)
    
    def eos_tensor(self, device: str=None):
        return self.list2tensor([self.eos_id()], device=device)
    
    def pad_tensor(self, device: str=None):
        return self.list2tensor([self.pad_id()], device=device)
    
    def unk_tensor(self, device: str=None):
        return self.list2tensor([self.unk_id()], device=device)
    
    def save(self, file: str):
        with open(file, mode="wb") as fo:
            pickle.dump(self, fo)

    def load(file: str) -> Self:
        with open(file, "rb") as f:
            lang = pickle.load(f)
        return lang

class DynamicLanguage(Language):
    """Language that constructs vocabulary from dataset"""
    def __init__(self):
        self._vocab: list[str] = []
        self._token2id = {}
        self._id2token = {}

    @abstractmethod
    def sentence2tokens(self, sentence: str, include_eos: bool=True) -> list[str]:
        """Split sentence to token strs, should include bos"""
        pass
    
    # implement
    def sentence2ids(self, sentence: str, include_eos: bool=True) -> list[int]:
        return [self.token2id(tok) for tok in self.sentence2tokens(sentence, include_eos=include_eos)]

    # implement
    def ids2sentence(self, ids: list[int]) -> str:
        # remove bos and eos
        ids = ids[1:]
        if ids and ids[-1] == self.eos_id():
            ids = ids[:-1]
        return "".join(self.id2token(i) for i in ids)

    def build_vocab(self, splits: dict[str, list[dict]], key="text"):
        """splits: can be dataset (ds)"""
        counter = Counter()
        for _, examples in splits.items():
            for ex in examples:
                tokens = self.sentence2tokens(ex[key])
                counter.update(tokens)
        self._vocab = sorted(counter.keys())
        self._vocab.append(self.pad_token())
        self._vocab.append(self.unk_token())
        self._token2id = {tok: id for id, tok in enumerate(self._vocab)}
        self._id2token = {id: tok for tok, id in self._token2id.items()}

    # implement
    def vocab(self):
        return self._vocab

    # implement
    def token2id(self, token: str) -> int:
        return self._token2id.get(token, self._token2id[self.unk_token()])

    # implement
    def id2token(self, token_id: int) -> str:
        return self._id2token[token_id]
    
    @classmethod
    def load_tokens_list(cls, tokens: list[str]) -> Self:
        lang = cls()
        lang._vocab = tokens

        for i, t in enumerate(tokens):
            lang._token2id[t] = i
            lang._id2token[i] = t
        
        return lang

class MolLanguage(Language):
    @abstractmethod
    def sentence2ids(self, sentence: str) -> list[int]:
        """Convert sentence to token ids"""
        pass
    
    @abstractmethod
    def token2id(self, token: str) -> int:
        pass

    @abstractmethod
    def id2token(token_id: int) -> str:
        pass

    @abstractmethod
    def vocab(self) -> list[str]:
        """List of all possible tokens. Can be dynamic (thus not a static method)"""
        pass

    @abstractmethod
    def ids2sentence(self, ids: list[int]) -> str:
        """Revert the token id sequence to sentence"""
        pass
    
    @abstractmethod
    def sentence2mol(self, sentence: str) -> Mol:
        pass

# Should be (DynamicLanguage, MolConvertibleLanguage) for MRO
class DynamicMolLanguage(DynamicLanguage, MolLanguage):
    @abstractmethod
    def sentence2tokens(self, sentence: str) -> list[str]:
        """Split sentence to token strs, should include bos and eos"""
        pass

    @abstractmethod
    def sentence2mol(self, sentence: str) -> Mol:
        pass