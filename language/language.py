from abc import ABC, abstractmethod
from rdkit.Chem import Mol
import torch


#vocabulary can be dynamic for better compatibility, thus most methods are not static
class Language(ABC):
  _bos_token = "<BOS>"
  _eos_token = "<EOS>"
  _pad_token = "<PAD>"
  _unk_token = "<UNKNOWN>"

  @abstractmethod
  #convert sentence to token ids, used for training
  def sentence2ids(self, sentence: str) -> list[int]:
    raise NotImplementedError("sentence2ids needs to be implemented.")
  
  @abstractmethod
  def token2id(self, token: str) -> int:
    raise NotImplementedError("token2id needs to be implemented.")

  @abstractmethod
  def id2token(tokenid: int) -> str:
    raise NotImplementedError("id2token needs to be implemented.")

  @abstractmethod
  #list of all possible tokens, can be dynamic (thus not a static method)
  def vocab(self) -> list[str]:
    raise NotImplementedError("vocab needs to be implemented.")

  @abstractmethod
  #revert the token id sequence to sentence
  def ids2sentence(self, idseq: list[int]) -> str:
    raise NotImplementedError("ids2sentence needs to be implemented.")
  
  @staticmethod
  @abstractmethod
  def sentence2mol(sentence: str) -> Mol:
    raise NotImplementedError("sentence2mol needs to be implemented.")
  
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
