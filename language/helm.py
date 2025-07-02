from rdkit import Chem
from rdkit.Chem import Mol
from language import DynamicMolLanguage
from utils import HELMConverter

# TODO: rewrite without using self.backbone_monomer_ids
class HELM(DynamicMolLanguage):
    def __init__(self, has_period=False, converter: HELMConverter=None):
        self.has_period = has_period
        self.backbone_monomer_ids = set()
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
        raw_tokenids = [self.token2idx(tok) for tok in self.sentence2tokens(sentence, include_eos=include_eos)]
        if self.has_period:
            return raw_tokenids

        noperiod_tokenids = []
        for i, tokenid in enumerate(raw_tokenids):
            if tokenid == self.token2idx("."):
                # index conditions shouldn't be needed for valid helm sentence
                if i > 0:
                    self.backbone_monomer_ids.add(raw_tokenids[i-1])
                if i < len(raw_tokenids) - 1:
                    self.backbone_monomer_ids.add(raw_tokenids[i+1])
            else:
                noperiod_tokenids.append(tokenid)

        return noperiod_tokenids
    
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
            newidseq = []
            for i, tokenid in enumerate(indices):
                newidseq.append(tokenid)
                if i < len(indices) - 1:
                    if indices[i] in self.backbone_monomer_ids and indices[i+1] in self.backbone_monomer_ids:
                        newidseq.append(self.token2idx("."))
            s = "".join(self.idx2token(i) for i in newidseq)
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