from typing import Self
import torch
from rdkit.Chem import Mol
from language import Language, MolLanguage
from node import Node, MolNode

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
    return self.__class__(idtensor=torch.cat([self.idtensor, Language.list2tensor([id])], dim=1), lang=self.lang, parent=self, lastprob=prob)

  #override
  def is_terminal(self):
    return self.idtensor[0][-1] == self.lang.eos_id()

  #output token id sequence as a list
  def idlist(self) -> list[int]:
    return self.idtensor[0].tolist()

  #bos node, often used as root
  @classmethod
  def bos_node(cls, lang: Language) -> Self:
    return cls(idtensor = lang.bos_tensor(), lang=lang)

class MolSentenceNode(SentenceNode, MolNode):
  def __init__(self, idtensor: torch.Tensor, lang: MolLanguage, parent=None, lastprob=1.0):
    self._is_valid_mol = None
    super().__init__(idtensor, lang, parent, lastprob)  

  #override
  def mol(self) -> Mol:
    mol = self.lang.__class__.sentence2mol(self.__str__())
    self._is_valid_mol = not (mol is None or mol.GetNumAtoms()==0)
    return mol