import pickle
import re
from typing import Self
from rdkit import Chem
from rdkit.Chem import Mol
import selfies
from language import DynamicMolLanguage

class SELFIES(DynamicMolLanguage):
    # implement
    def sentence2tokens(self, sentence: str, include_eos: bool=True) -> list[str]:
        tokens = list(selfies.split_selfies(sentence)) # raises selfies.DecoderError if failed
        tokens.insert(0, self.bos_token())
        if include_eos:
            tokens.append(self.eos_token())
        return tokens

    # implement
    def sentence2mol(self, sentence: str) -> Mol:
        try:
            smiles = selfies.decoder(sentence)
            return Chem.MolFromSmiles(smiles)
        except selfies.EncoderError:
            return None