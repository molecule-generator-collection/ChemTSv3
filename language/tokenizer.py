import torch
from language import Language

class TokenizerLanguage(Language):
    """Language class wrapping a Hugging Face Transformers Tokenizer."""
    def __init__(self, tokenizer, device: str=None):
        self.tokenizer = tokenizer
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._bos_token = tokenizer.bos_token
        self._eos_token = tokenizer.eos_token
        self._pad_token = tokenizer.pad_token
        self._unk_token = tokenizer.unk_token
        self._vocab = self._make_vocab()
        
    def sentence2ids(self, sentence: str, include_eos: bool=True) -> list[int]:
        ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        if include_eos:
            if not ids or ids[-1] != self.eos_id():
                ids = ids + [self.eos_id()]
        return ids
    
    def token2id(self, token: str) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        if tid is None:
            # convert_tokens_to_ids can return None for unknown?
            if self.unk_id() is not None:
                return self.unk_id()
            raise KeyError(f"Token not in vocab and no unk_token_id: {token!r}")
        return tid

    def id2token(self, tokenid: int) -> str:
        tok = self.tokenizer.convert_ids_to_tokens(int(tokenid), skip_special_tokens=False)
        if tok is None:
            raise KeyError(f"Invalid token id: {tokenid}")
        return tok
    
    def _make_vocab(self):
        vdict = self.tokenizer.get_vocab()
        by_id = sorted(vdict.items(), key=lambda kv: kv[1])
        return [tok for tok, _id in by_id]

    def vocab(self) -> list[str]:
        return self._vocab

    def ids2sentence(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)