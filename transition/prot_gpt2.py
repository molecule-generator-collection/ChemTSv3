import logging
from transformers import pipeline
from language import FASTA
from node import FASTAStringNode
from transition import Transition

class ProtGPT2Transition(Transition):
    CANDIDATES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", # 1~10
             "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", # 11~20
             "W", "Y", "Z", "<EOS>"]
    
    def __init__(self, rollout_top_k: int=950, max_length: int =100, logger: logging.Logger=None):
        self.protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2", device="cuda")
        self.rollout_top_k = rollout_top_k
        self._max_length = max_length
        super().__init__(logger=logger)
        
    def max_length(self):
        return self._max_length
        
    def next_nodes(self, node: FASTAStringNode):
        children = []
        parent = node.string
        if len(parent) > self.max_length() or parent.endswith("<EOS>"):
            return []
        
        for a in self.CANDIDATES:
            new_string = parent + a
            children.append(FASTAStringNode(new_string, parent=parent, last_prob=1/len(self.CANDIDATES), last_action=a))
        
        return children
    
    def rollout(self, initial_node: FASTAStringNode):
        initial_string = initial_node.string
        
        self.protgpt2("<|endoftext|>" + initial_string, max_length=self._max_length, do_sample=True, top_k=self.rollout_top_k, num_return_sequences=1, eos_token_id=0)