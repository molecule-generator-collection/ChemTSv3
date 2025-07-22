from typing import Self, Any
from rdkit.Chem import Mol
from language import MolLanguage, SMILES
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
    def node_from_key(cls, key: str, lang: MolLanguage=None, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        return MolStringNode(string=key, lang=lang, parent=parent, last_prob=last_prob, last_action=last_action)
    
    # implement
    def _mol_impl(self) -> Mol:
        return self.lang.sentence2mol(self.string)
    
class SMILESStringNode(MolStringNode):
    smiles_lang = SMILES()
    
    def __init__(self, string: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        super().__init__(string=string, lang=self.smiles_lang, parent=parent, last_prob=last_prob, last_action=last_action)
    
    @classmethod
    def node_from_key(cls, key: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        return SMILESStringNode(string=key, parent=parent, last_prob=last_prob, last_action=last_action)
    
    # override
    def smiles(self, use_cache=False) -> str:
        """Expects self.string to be canonical SMILES."""
        return self.string
    
class SELFIESStringNode(MolStringNode):
    from language import SELFIES # lazy import
    selfies_lang = SELFIES()
    
    def __init__(self, string: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        super().__init__(string=string, lang=self.selfies_lang, parent=parent, last_prob=last_prob, last_action=last_action)
    
    @classmethod
    def node_from_key(cls, key: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        return SELFIESStringNode(string=key, parent=parent, last_prob=last_prob, last_action=last_action)