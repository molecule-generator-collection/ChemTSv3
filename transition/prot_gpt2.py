import logging
from transformers import pipeline
from language import FASTA
from node import FASTAStringNode
from transition import Transition

class ProtGPT2Transition(Transition):
    def __init__(self, logger: logging.Logger=None):
        self.protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2", device="cuda")
        super().__init__(logger=logger)
        
    def next_nodes(self, node: FASTAStringNode):
        parent = node.string
        for a in FASTA.PROTEIN_TOKENS:
            pass