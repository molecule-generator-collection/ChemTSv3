from typing import Self, Any
from rdkit import Chem
from rdkit.Chem import Mol
import torch
from language import Language, MolLanguage, SMILES
from node import Node, MolNode
from utils import mol_validity_check

class SentenceNode(Node):
    lang: Language = None
    device = None
    
    def __init__(self, id_tensor: torch.Tensor, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        self.id_tensor = id_tensor
        super().__init__(parent=parent, last_prob=last_prob, last_action=last_action)

    # implement
    def key(self):
        return self.lang.ids2sentence(self.id_list())

    # implement
    def has_reward(self):
        return self.id_tensor[0][-1] == self.lang.eos_id()
    
    # implement
    @classmethod
    def node_from_key(cls, key: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None, include_eos: bool=False) -> Self:
        id_tensor = cls.lang.sentence2tensor(key, include_eos=include_eos, device=cls.device)
        return cls(id_tensor=id_tensor, parent=parent, last_prob=last_prob, last_action=last_action)

    def id_list(self) -> list[int]:
        """Output token id sequence as a list"""
        return self.id_tensor[0].tolist()
    
    # override
    def discard_unneeded_states(self):
        """Clear states no longer needed after transition to reduce memory usage."""
        self.id_tensor = None

class MolSentenceNode(SentenceNode, MolNode):
    use_canonical_smiles_as_key: bool = False
    lang: MolLanguage = None
    device = None

    def __init__(self, id_tensor: torch.Tensor, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        super().__init__(id_tensor=id_tensor, parent=parent, last_prob=last_prob, last_action=last_action)

    # implement
    def _mol_impl(self) -> Mol:
        return self.lang.sentence2mol(self.lang.ids2sentence(self.id_list()))

    # override     
    def key(self):
        if not self.use_canonical_smiles_as_key:
            return super().key()
        mol = self.mol(use_cache=False)
        if not mol_validity_check(mol):
            return super().key()
        else:
            try:
                return Chem.MolToSmiles(mol, canonical=True)
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