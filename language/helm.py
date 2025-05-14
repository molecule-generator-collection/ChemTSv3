from language import DynamicMolLanguage
import re
from rdkit import Chem
from rdkit.Chem import Mol
import pickle
from typing import Self

class HELM(DynamicMolLanguage):
  #Currently has_period = True isn't properly implemented for general use
  #override
  def __init__(self, has_period=False):
    self.has_period = has_period
    self.monomer_ids = set()
    super().__init__()

  #override
  def sentence2tokens(self, sentence):
    helm = HELM.eos_culling(sentence)

    #pattern by Shoichi Ishida
    pattern = "(\[[^\]]+]|PEPTIDE[0-9]+|RNA[0-9]+|CHEM[0-9]+|BLOB[0-9]+|R[0-9]|A|C|D|E|F|G|H|I|K|L|M|N|P|Q|R|S|T|V|W|Y|\||\(|\)|\{|\}|-|\$|:|,|\.|[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [t for t in regex.findall(helm)]
    assert helm == "".join(tokens)

    tokens.insert(0, self.bos_token())
    tokens.append(self.eos_token())
    
    return tokens

  #override
  def sentence2ids(self, sentence):
    raw_tokenids = [self.token2id(tok) for tok in self.sentence2tokens(sentence)]
    if self.has_period:
      return raw_tokenids

    noperiod_tokenids = []
    for i, tokenid in enumerate(raw_tokenids):
      if tokenid == self.token2id("."):
        #index conditions shouldn't be needed, but implemented for precaution
        if i > 0:
          self.monomer_ids.add(raw_tokenids[i-1])
        if i < len(raw_tokenids) - 1:
          self.monomer_ids.add(raw_tokenids[i+1])
      else:
        noperiod_tokenids.append(tokenid)

    return noperiod_tokenids
  
  #override
  def ids2sentence(self, idseq):
    idseq = idseq[1:-1]
    #add periods
    if not self.has_period:
      newidseq = []
      for i, tokenid in enumerate(idseq):
        newidseq.append(tokenid)
        if i < len(idseq) - 1:
          if idseq[i] in self.monomer_ids and idseq[i+1] in self.monomer_ids:
            newidseq.append(self.token2id("."))
      s = "".join(self.id2token(i) for i in newidseq)
    else:
      s = "".join(self.id2token(i) for i in idseq)
    s += "$$$$"
    return s

  @staticmethod    
  def eos_culling(sentence: str) -> str:
    if sentence.endswith('$$$$'):
        return sentence[:-4]
    elif sentence.endswith('$$$'):
        return sentence[:-3]
    elif sentence.endswith('$$'):
        return sentence[:-2]
    elif sentence.endswith('$'):
        return sentence[:-1]
    else:
        return sentence
    
  #override
  @staticmethod
  def sentence2mol(sentence: str) -> Mol:
    return Chem.MolFromHELM(sentence)
  
  #override
  def save(self, file: str):
    with open(file, mode="wb") as fo:
      pickle.dump(self._vocab, fo)
      pickle.dump(self._token2id, fo)
      pickle.dump(self._id2token, fo)
      pickle.dump(self.has_period, fo)
      pickle.dump(self.monomer_ids, fo)

  #override
  @staticmethod    
  def load(file: str) -> Self:
    lang = HELM()
    with open(file, "rb") as f:
      lang._vocab = pickle.load(f)
      lang._token2id = pickle.load(f)
      lang._id2token = pickle.load(f)
      lang.has_period = pickle.load(f)
      lang.monomer_ids = pickle.load(f)
    return lang