import pickle
import re
from typing import Self
from rdkit import Chem
from rdkit.Chem import Mol
from language import DynamicMolLanguage

class SMILES(DynamicMolLanguage):
    #implement
    def sentence2tokens(self, sentence):
        #Pattern from ChemTSv2: modified by Shoichi Ishida based on https://github.com/pschwllr/MolecularTransformer#pre-processing
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(sentence)]
        assert sentence == ''.join(tokens)

        tokens.insert(0, self.bos_token())
        tokens.append(self.eos_token())
        
        return tokens

    #implement
    @staticmethod
    def sentence2mol(sentence: str) -> Mol:
        return Chem.MolFromSmiles(sentence)