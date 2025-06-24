from typing import Self, Any
from rdkit import Chem
from rdkit.Chem import Mol
import torch
from language import Language, MolLanguage
from node import Node, MolNode

class SentenceNode(Node):
    def __init__(self, id_tensor: torch.Tensor, lang: Language, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        self.id_tensor = id_tensor
        self.lang = lang
        super().__init__(parent=parent, last_prob=last_prob, last_action=last_action)

    # implement
    def __str__(self):
        return self.lang.ids2sentence(self.id_list())

    # implement
    def is_terminal(self):
        return self.id_tensor[0][-1] == self.lang.eos_id()
    
    # implement
    @classmethod
    def node_from_str(cls, lang: Language, string: str, include_eos: bool=False, device: str=None, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        id_tensor = lang.sentence2tensor(string, include_eos=include_eos, device=device)
        return cls(id_tensor=id_tensor, lang=lang, parent=parent, last_prob=last_prob, last_action=last_action)

    def id_list(self) -> list[int]:
        """output token id sequence as a list"""
        return self.id_tensor[0].tolist()

    @classmethod
    def bos_node(cls, lang: Language, device: str=None) -> Self:
        """make bos node, often used as root"""
        return cls(id_tensor = lang.bos_tensor(device), lang=lang)

class MolSentenceNode(SentenceNode, MolNode):
    def __init__(self, id_tensor: torch.Tensor, lang: MolLanguage, parent=None, last_prob=1.0, last_action=None):
        super().__init__(id_tensor=id_tensor, lang=lang, parent=parent, last_prob=last_prob, last_action=last_action)

    # implement
    def _mol_impl(self) -> Mol:
        return self.lang.sentence2mol(self.lang.ids2sentence(self.id_list()))