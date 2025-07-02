import logging
from transformers import AutoTokenizer, T5ForConditionalGeneration
from node import MolStringNode
from transition import BlackBoxTransition

TOKENIZER = AutoTokenizer.from_pretrained("QizhiPei/biot5-plus-base-mol-instructions-molecule")
MODEL = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-plus-base-mol-instructions-molecule')

class BioT5Transition(BlackBoxTransition):
    def __init__(self, target_objective: str, prompt_prefix: str=None, n_samples=2, logger: logging.Logger=None):
        self.target_objective = target_objective
        self.prompt_prefix = prompt_prefix or "Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that "
        super().__init__(logger=logger)
        
    def sample_transition(self, node: MolStringNode) -> MolStringNode:
        sel = node.string
        prompt = self.prompt_prefix + self.target_objective + ". Now complete the following example - Input: <bom>" + sel + "<eom> Output: "
        
        input_ids = TOKENIZER(prompt, return_tensors="pt").input_ids
        outputs = MODEL.generate(input_ids, max_length=512, do_sample=True)
        output_selfies = TOKENIZER.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
        
        return MolStringNode(string=output_selfies, lang=node.lang, parent=node)