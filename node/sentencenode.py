from typing import Self
import torch
from rdkit.Chem import Mol
from language.language import Language, MolConvertibleLanuguage
from .node import Node, MolConvertibleNode

class SentenceNode(Node):
  def __init__(self, idtensor: torch.Tensor, lang: Language, parent=None, lastprob=1.0):
    self.idtensor = idtensor
    self.lang = lang
    super().__init__(parent=parent, lastprob=lastprob)

  #override
  def __str__(self):
    return self.lang.ids2sentence(self.idlist())
    pass

  #override
  #should be ordered by token id
  #doesn't register nodes as children
  def nextnodes(self):
    return [self.nextnode(id) for id in range(len(self.lang.vocab()))]

  def nextnode(self, id: int, prob: float = 1.0) -> Self:
    return SentenceNode(idtensor=torch.cat([self.idtensor, Language.list2tensor([id])], dim=1), lang=self.lang, parent=self, lastprob=prob)

  #override
  def is_terminal(self):
    return self.idtensor[0][-1] == self.lang.eos_id()

  #output token id sequence as a list
  def idlist(self) -> list[int]:
    return self.idtensor[0].tolist()

  #bos node, often used as root
  @staticmethod
  def bos_node(lang: Language) -> Self:
    return SentenceNode(idtensor = lang.bos_tensor(), lang=lang)

class MolConvertibleSentenceNode(SentenceNode, MolConvertibleNode):
  def __init__(self, idtensor: torch.Tensor, lang: MolConvertibleLanuguage, parent=None, lastprob=1.0):
    self.idtensor = idtensor
    self.lang = lang
    super().__init__(parent=parent, lastprob=lastprob)  

  #override
  def mol(self) -> Mol:
    return self.lang.__class__.sentence2mol(self.__str__())