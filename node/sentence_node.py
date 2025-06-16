from typing import Self
from rdkit import Chem
from rdkit.Chem import Mol
import torch
from language import Language, MolLanguage
from node import Node, MolNode

class SentenceNode(Node):
    def __init__(self, id_tensor: torch.Tensor, lang: Language, parent=None, last_prob=1.0):
        self.id_tensor = id_tensor
        self.lang = lang
        super().__init__(parent=parent, last_prob=last_prob)

    # implement
    def __str__(self):
        return self.lang.ids2sentence(self.id_list())

    # implement
    def is_terminal(self):
        return self.id_tensor[0][-1] == self.lang.eos_id()

    # output token id sequence as a list
    def id_list(self) -> list[int]:
        return self.id_tensor[0].tolist()

    # bos node, often used as root
    @classmethod
    def bos_node(cls, lang: Language) -> Self:
        return cls(id_tensor = lang.bos_tensor(), lang=lang)

class MolSentenceNode(SentenceNode, MolNode):
    use_canonical_smiles = True    

    def __init__(self, id_tensor: torch.Tensor, lang: MolLanguage, parent=None, last_prob=1.0):
        self._canonical_smiles = None
        super().__init__(id_tensor, lang, parent, last_prob)

    # implement
    def _mol_impl(self) -> Mol:
        return self.lang.sentence2mol(self.__str__())
    
    #override
    def __str__(self):
        if self.use_canonical_smiles:
            if self._canonical_smiles is None:
                self._canonical_smiles = Chem.MolToSmiles(self.mol())
            return self._canonical_smiles
        else:
            return super().__str__()