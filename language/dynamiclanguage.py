
from abc import abstractmethod
from .language import Language
from collections import Counter

#language that makes vocabulary from dataset
class DynamicLanguage(Language):
  def __init__(self):
    self._vocab: list[str] = []
    self._token2id = {}
    self._id2token = {}

  #split sentence to token strs, should include bos and eos
  @abstractmethod
  def sentence2tokens(self, sentence: str) -> list[str]:
    raise NotImplementedError("sentence2tokens needs to be implemented.")
  
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

  #override
  def vocab(self):
    return self._vocab

  #override
  def token2id(self, token):
    return self._token2id.get(token, self._token2id[self.unk_token()])

  #override
  def id2token(self, tokenid):
    return self._id2token[tokenid]