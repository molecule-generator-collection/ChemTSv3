from typing import Self
import torch
from rdkit.Chem import Mol
from language import Language, MolLanguage
from node import Node, MolNode

class SentenceNode(Node):
  def __init__(self, id_tensor: torch.Tensor, lang: Language, parent=None, last_prob=1.0):
    self.id_tensor = id_tensor
    self.lang = lang
    super().__init__(parent=parent, last_prob=last_prob)

  #override
  def __str__(self):
    return self.lang.ids2sentence(self.id_list())
    pass

  #override
  #should be ordered by token id
  #doesn't register nodes as children
  def child_candidates(self):
    return [self.child_candidate(id) for id in range(len(self.lang.vocab()))]

  def child_candidate(self, id: int, prob: float=1.0) -> Self:
    return self.__class__(id_tensor=torch.cat([self.id_tensor, Language.list2tensor([id])], dim=1), lang=self.lang, parent=self, last_prob=prob)

  #override
  def is_terminal(self):
    return self.id_tensor[0][-1] == self.lang.eos_id()

  #output token id sequence as a list
  def id_list(self) -> list[int]:
    return self.id_tensor[0].tolist()

  #bos node, often used as root
  @classmethod
  def bos_node(cls, lang: Language) -> Self:
    return cls(id_tensor = lang.bos_tensor(), lang=lang)

class MolSentenceNode(SentenceNode, MolNode):
  def __init__(self, id_tensor: torch.Tensor, lang: MolLanguage, parent=None, last_prob=1.0):
    self._is_valid_mol = None
    super().__init__(id_tensor, lang, parent, last_prob)  

  #override
  def mol(self) -> Mol:
    mol = self.lang.__class__.sentence2mol(self.__str__())
    self._is_valid_mol = not (mol is None or mol.GetNumAtoms()==0)
    return mol