from rdkit import Chem
from rdkit.Chem import Mol
from language import MolLanguage

class FASTA(MolLanguage):
    TOKENS = ["<BOS>", # 0
             "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", # 1~10
             "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", # 11~20
             "W", "Y", "Z", "X", "*", "-", # 21~26
             "<EOS>", "<PAD>"] # 27, 28
    _TOKEN2ID = {tok: i for i, tok in enumerate(TOKENS)}
    _unk_token = "X" # override
    
    def __init__(self, flavor: int=0):
        self.flavor = flavor # Used in MolFromFASTA()
    
    def sentence2ids(self, sentence: str, include_eos: bool=True) -> list[int]:
        """Convert sentence to token ids"""
        ids = [self.token2id("<BOS>")]
        ids.extend(self.token2id(ch) for ch in sentence)
        if include_eos:
            ids.append(self.token2id("<EOS>"))
        return ids
    
    def token2id(self, token: str) -> int:
        if token not in self._TOKEN2ID:
            token = self._unk_token
        return self._TOKEN2ID[token]

    def id2token(token_id: int) -> str:
        if 0 <= token_id <= 28:
            return FASTA.TOKENS[token_id]
        else:
            return FASTA._unk_token

    def vocab(self) -> list[str]:
        """List of all possible tokens."""
        return self.TOKENS

    def ids2sentence(self, ids: list[int]) -> str:
        """Revert the token id sequence to sentence"""
        specials = {"<BOS>", "<EOS>", "<PAD>"}
        chars = []
        for i in ids:
            tok = FASTA.id2token(i)
            if tok in specials:
                continue
            chars.append(tok)
        return "".join(chars)
    
    def sentence2mol(self, sentence: str) -> Mol:
        try:
            return Chem.MolFromFASTA(sentence, flavor=self.flavor)
        except:
            return None