from typing import Self, Any
from rdkit import Chem
from rdkit.Chem import Mol
import torch
from language import Language, MolLanguage, SMILES
from node import Node, MolNode

class SentenceNode(Node):
    def __init__(self, id_tensor: torch.Tensor, lang: Language, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        self.id_tensor = id_tensor
        self.lang = lang
        super().__init__(parent=parent, last_prob=last_prob, last_action=last_action)

    # implement
    def key(self):
        return self.lang.ids2sentence(self.id_list())

    # implement
    def has_reward(self):
        return self.id_tensor[0][-1] == self.lang.eos_id()
    
    # implement
    @classmethod
    def node_from_key(cls, key: str, lang: Language, include_eos: bool=False, device: str=None, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        id_tensor = lang.sentence2tensor(key, include_eos=include_eos, device=device)
        return cls(id_tensor=id_tensor, lang=lang, parent=parent, last_prob=last_prob, last_action=last_action)

    def id_list(self) -> list[int]:
        """Output token id sequence as a list"""
        return self.id_tensor[0].tolist()

    @classmethod
    def bos_node(cls, lang: Language, device: str=None) -> Self:
        """Make bos node. Often used as root."""
        return cls.node_from_key(key="", lang=lang, device=device, include_eos=False)
    
    # override
    def discard_unneeded_states(self):
        """Clear states no longer needed after transition to reduce memory usage."""
        self.id_tensor = None
        self.lang = None

class MolSentenceNode(SentenceNode, MolNode):
    use_canonical_smiles_as_key = False

    def __init__(self, id_tensor: torch.Tensor, lang: MolLanguage, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        super().__init__(id_tensor=id_tensor, lang=lang, parent=parent, last_prob=last_prob, last_action=last_action)

    # implement
    def _mol_impl(self) -> Mol:
        return self.lang.sentence2mol(self.lang.ids2sentence(self.id_list()))
    
    # override
    def key(self):
        if not self.use_canonical_smiles_as_key or not self.validity_filter().check(self):
            return super().key()
        else:
            try:
                return Chem.MolToSmiles(self.mol(use_cache=True), canonical=True)
            except Exception:
                return "invalid mol"
        
    # override
    def smiles(self, use_cache=False) -> str:
        if isinstance(self.lang, SMILES):
            return self.lang.tensor2sentence(self.id_tensor)
        else:
            return super().smiles(use_cache=use_cache)
        
    # override
    def discard_unneeded_states(self):
        """Clear states no longer needed after transition to reduce memory usage."""
        self.id_tensor = None
        self.lang = None