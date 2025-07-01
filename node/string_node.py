from typing import Self, Any
from rdkit import Chem
from rdkit.Chem import Mol
import torch
from language import Language, MolLanguage
from node import Node, MolNode

class MolStringNode(MolNode):
    def __init__(self, string: str, lang_class: type=None, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        self.string = string
        self.lang_class = lang_class
        super().__init__(parent=parent, last_prob=last_prob, last_action=last_action)
    
    # implement
    def key(self):
        return self.string
    
    