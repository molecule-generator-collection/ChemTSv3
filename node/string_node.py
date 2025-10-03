from typing import Self, Any
from rdkit import Chem
from rdkit.Chem import Mol
from language import MolLanguage, SMILES, FASTA
from node import Node, MolNode
from utils import mol_validity_check

class StringNode(Node):
    eos: str = None # Can be optionally set
    
    def __init__(self, string: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None):
        self.string = string
        super().__init__(parent=parent, last_prob=last_prob, last_action=last_action)
        
    def has_reward(self):
        if self.eos is None:
            return True
        else:
            if self.string.endswith(self.eos):
                return True
            else:
                return False
            
    def key(self):
        return self.string
    
    @classmethod
    def node_from_key(cls, key: str, parent: Self=None, last_prob: float=1.0, last_action: Any=None) -> Self:
        return cls(string=key, parent=parent, last_prob=last_prob, last_action=last_action)
    
    # override
    def discard_unneeded_states(self):
        """Clear states no longer needed after transition to reduce memory usage."""
        self.string = None

class MolStringNode(StringNode, MolNode):
    use_canonical_smiles_as_key: bool = False
    lang: MolLanguage = None

    # override
    def key(self):
        raw_key = self.string if self.eos is None else self.string.replace(self.eos, "")

        if not self.use_canonical_smiles_as_key:
            return raw_key
        mol = self.mol(use_cache=False) # TODO: allow =True in some cases
        if not mol_validity_check(mol):
            return raw_key
        else:
            try:
                return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                return "invalid mol"

    def _mol_impl(self) -> Mol:
        if self.eos is None:
            return self.lang.sentence2mol(self.string)
        else:
            return self.lang.sentence2mol(self.string.replace(self.eos, ""))
    
class SMILESStringNode(MolStringNode):
    lang = SMILES()
    
    # override
    def smiles(self, use_cache=False) -> str:
        """Expects self.string to be canonical SMILES."""
        return self.string
    
class CanonicalSMILESStringNode(SMILESStringNode):
    # override
    def key(self):
        return self.string
    
class FASTAStringNode(MolStringNode):
    flavor = 0
    lang = FASTA(flavor)