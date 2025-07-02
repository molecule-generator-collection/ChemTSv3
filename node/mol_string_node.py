from typing import Self, Any
from rdkit.Chem import Mol
from language import MolLanguage
from node import MolNode

class MolStringNode(MolNode):
    def __init__(self, string: str, lang: MolLanguage=None, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        self.string = string
        self.lang = lang
        super().__init__(parent=parent, last_prob=last_prob, last_action=last_action)
    
    # implement
    def has_reward(self):
        return True
    
    # implement
    def key(self):
        return self.string
    
    @classmethod
    def node_from_key(cls, key: str, lang: MolLanguage=None, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        return MolStringNode(string=key, lang=lang, parent=parent, last_prob=last_prob, last_action=last_action)
    
    # implement
    def _mol_impl(self) -> Mol:
        return self.lang.sentence2mol(self.string)