from typing import Self, Any
import selfies
from node import MolStringNode
    
class SELFIESStringNode(MolStringNode):
    from language import SELFIES # lazy import
    lang = SELFIES()
    
    # implement
    @classmethod
    def node_from_key(cls, key: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        if cls.use_canonical_smiles_as_key:
            try:
                sel = selfies.encoder(key)
                return SELFIESStringNode(string=sel, parent=parent, last_prob=last_prob, last_action=last_action)
            except:
                return SELFIESStringNode(string=key, parent=parent, last_prob=last_prob, last_action=last_action)
        else:
            return SELFIESStringNode(string=key, parent=parent, last_prob=last_prob, last_action=last_action)