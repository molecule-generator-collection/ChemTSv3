from abc import ABC, abstractmethod
import torch
from rdkit.Chem import Mol
from collections import Counter
import pickle
from typing import Self

#vocabulary can be dynamic for better compatibility, thus most methods are not static
class Language(ABC):
  _bos_token = "<BOS>"
  _eos_token = "<EOS>"
  _pad_token = "<PAD>"
  _unk_token = "<UNKNOWN>"

  @abstractmethod
  #convert sentence to token ids, used for training
  def sentence2ids(self, sentence: str) -> list[int]:
    pass
  
  @abstractmethod
  def token2id(self, token: str) -> int:
    pass

  @abstractmethod
  def id2token(tokenid: int) -> str:
    pass

  @abstractmethod
  #list of all possible tokens, can be dynamic (thus not a static method)
  def vocab(self) -> list[str]:
    pass

  @abstractmethod
  #revert the token id sequence to sentence
  def ids2sentence(self, idseq: list[int]) -> str:
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
  
  @staticmethod
  def list2tensor(li: list[int]) -> torch.Tensor:
    return torch.tensor([li], device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
  
  @staticmethod
  def tensor2list(t: torch.Tensor) -> list[int]:
    return t[0].tolist()
  
  def bos_tensor(self):
    return Language.list2tensor([self.bos_id()])
  
  def eos_tensor(self):
    return Language.list2tensor([self.eos_id()])
  
  def pad_tensor(self):
    return Language.list2tensor([self.pad_id()])
  
  def unk_tensor(self):
    return Language.list2tensor([self.unk_id()])

#language that makes vocabulary from dataset
class DynamicLanguage(Language):
  def __init__(self):
    self._vocab: list[str] = []
    self._token2id = {}
    self._id2token = {}

  #split sentence to token strs, should include bos and eos
  @abstractmethod
  def sentence2tokens(self, sentence: str) -> list[str]:
    pass
  
  #implement
  def sentence2ids(self, sentence):
    return [self.token2id(tok) for tok in self.sentence2tokens(sentence)]

  #implement
  def ids2sentence(self, idseq):
    idseq = idseq[1:-1] #remove bos and eos
    return "".join(self.id2token(i) for i in idseq)

  #can input dataset
  def build_vocab(self, splits: dict[str, list[dict]]):
    counter = Counter()
    for split_name, examples in splits.items():
        for ex in examples:
            tokens = self.sentence2tokens(ex["text"])
            counter.update(tokens)
    self._vocab = sorted(counter.keys())
    self._vocab.append(self.pad_token())
    self._vocab.append(self.unk_token())
    self._token2id = {tok: idx for idx, tok in enumerate(self._vocab)}
    self._id2token = {idx: tok for tok, idx in self._token2id.items()}

  #implement
  def vocab(self):
    return self._vocab

  #implement
  def token2id(self, token):
    return self._token2id.get(token, self._token2id[self.unk_token()])

  #implement
  def id2token(self, tokenid):
    return self._id2token[tokenid]
  
  def save(self, file: str):
    with open(file, mode="wb") as fo:
      pickle.dump(self._vocab, fo)
      pickle.dump(self._token2id, fo)
      pickle.dump(self._id2token, fo)

  @staticmethod    
  def load(file: str) -> Self:
    with open(file, "rb") as f:
      vocab_tmp = pickle.load(f)
      token2id_tmp = pickle.load(f)
      id2token_tmp = pickle.load(f)
    lang = DynamicLanguage()
    lang._vocab = vocab_tmp
    lang._token2id = token2id_tmp
    lang._id2token = id2token_tmp
    return lang

class MolLanguage(Language):
  @abstractmethod
  #convert sentence to token ids, used for training
  def sentence2ids(self, sentence: str) -> list[int]:
    pass
  
  @abstractmethod
  def token2id(self, token: str) -> int:
    pass

  @abstractmethod
  def id2token(tokenid: int) -> str:
    pass

  @abstractmethod
  #list of all possible tokens, can be dynamic (thus not a static method)
  def vocab(self) -> list[str]:
    pass

  @abstractmethod
  #revert the token id sequence to sentence
  def ids2sentence(self, idseq: list[int]) -> str:
    pass
  
  @staticmethod
  @abstractmethod
  def sentence2mol(sentence: str) -> Mol:
    pass

#Should be (DynamicLanguage, MolConvertibleLanguage) for MRO
class DynamicMolLanguage(DynamicLanguage, MolLanguage):
  #split sentence to token strs, should include bos and eos
  @abstractmethod
  def sentence2tokens(self, sentence: str) -> list[str]:
    pass

  @staticmethod
  @abstractmethod
  def sentence2mol(sentence: str) -> Mol:
    pass