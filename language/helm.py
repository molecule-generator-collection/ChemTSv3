import pickle
import re
from typing import Self
from rdkit import Chem
from rdkit.Chem import Mol
from language import DynamicMolLanguage
from utils import HELMConverter

class HELM(DynamicMolLanguage):
    def __init__(self, has_period=False, converter: HELMConverter=None):
        self.has_period = has_period
        self.converter = converter
        super().__init__()
    
    def load_monomer_library(self, *args: str, culling=False):
        self.converter = HELMConverter().load(*args)
        if culling:
            self.converter.lib.cull(self.vocab())

    # implement
    def sentence2tokens(self, sentence: str, include_eos: bool=True) -> list[str]:
        helm = HELM.cull_postfix(sentence)

        tokens = HELMConverter.split_helm(helm)

        tokens.insert(0, self.bos_token())
        if include_eos:
            tokens.append(self.eos_token())
        
        return tokens

    # override
    def sentence2indices(self, sentence: str, include_eos: bool=True):
        token_indices = [self.token2idx(tok) for tok in self.sentence2tokens(sentence, include_eos=include_eos)]
        if self.has_period:
            return token_indices

        token_indices_without_period = []
        for tokenid in token_indices:
            if tokenid != self.token2idx("."):
                token_indices_without_period.append(tokenid)

        return token_indices_without_period
    
    # override
    def indices2sentence(self, indices: list[int]):
        if indices[0] == self.bos_id():
            if len(indices) == 1:
                return self.idx2token(indices[0])
            indices = indices[1:]
        if indices[-1] == self.eos_id():
            indices = indices[:-1]
        # add periods
        if not self.has_period:
            new_indices = []
            for i, tokenid in enumerate(indices):
                new_indices.append(tokenid)
                if i < len(indices) - 1:
                    if self.is_monomer_idx(indices[i]) and self.is_monomer_idx(indices[i+1]):
                        new_indices.append(self.token2idx("."))
            s = "".join(self.idx2token(i) for i in new_indices)
        else:
            s = "".join(self.idx2token(i) for i in indices)
        s += "$$$$"
        return s

    @staticmethod
    def cull_postfix(sentence: str) -> str:
        if sentence.endswith("V2.0"):
            sentence = sentence[:-4]
        if sentence.endswith("$$$$"):
            return sentence[:-4]
        elif sentence.endswith("$$$"):
            return sentence[:-3]
        elif sentence.endswith("$$"):
            return sentence[:-2]
        elif sentence.endswith("$"):
            return sentence[:-1]
        else:
            return sentence
        
    # implement
    def sentence2mol(self, sentence: str) -> Mol:
        if self.converter is None:
            return Chem.MolFromHELM(sentence)
        else:
            return self.converter.convert(sentence)
    
    @staticmethod
    def is_monomer_token(s: str) -> bool:
        return bool(re.fullmatch(r"[a-zA-Z]{1}|\[[^\]]*\]", s))
    
    def is_monomer_idx(self, idx: int) -> bool:
        return self.is_monomer_token(self.idx2token(idx))

    # override
    def save(self, file: str):
        # decompose the mol object for RDKit cross-version compatibility
        if self.converter:
            self.converter.lib.cap_group_mols = {key: None for key in self.converter.lib.cap_group_mols}
        with open(file, mode="wb") as fo:
            pickle.dump(self, fo)

    # override
    def load(file: str) -> Self:
        with open(file, "rb") as f:
            lang = pickle.load(f)
        if lang.converter:
            lang.converter.lib.cap_group_mols = {key: Chem.MolFromSmiles(key) for key in lang.converter.lib.cap_group_mols}
        return lang