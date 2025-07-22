from typing import Self, Any
from node import MolStringNode
    
class SELFIESStringNode(MolStringNode):
    from language import SELFIES # lazy import
    selfies_lang = SELFIES()
    
    def __init__(self, string: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        super().__init__(string=string, lang=self.selfies_lang, parent=parent, last_prob=last_prob, last_action=last_action)
    
    @classmethod
    def node_from_key(cls, key: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        return SELFIESStringNode(string=key, parent=parent, last_prob=last_prob, last_action=last_action)