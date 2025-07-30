from typing import Self, Any
from rdkit import Chem
from rdkit.Chem import Mol
from language import MolLanguage, SMILES
from node import MolNode

class MolStringNode(MolNode):
    use_canonical_smiles_as_key = False
    
    def __init__(self, string: str, lang: MolLanguage=None, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        self.string = string
        self.lang = lang
        super().__init__(parent=parent, last_prob=last_prob, last_action=last_action)
    
    # implement
    def has_reward(self):
        return True
    
    def key(self):
        if not self.use_canonical_smiles_as_key or not self.validity_filter().check(self):
            return self.string
        else:
            try:
                return Chem.MolToSmiles(self.mol(use_cache=True), canonical=True)
            except Exception:
                return "invalid mol"
    
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