import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from node import MolStringNode
from transition import BlackBoxTransition

class BioT5Transition(BlackBoxTransition):
    def __init__(self, target_objective: str, prompt_prefix: str=None, prompt_postfix: str=None, n_samples=2, logger: logging.Logger=None):
        self.target_objective = target_objective
        self.prompt_prefix = prompt_prefix or "Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that"
        self.prompt_postfix = prompt_postfix or ". Now complete the following example"
        super().__init__(logger=logger)
        
        self.logger.info("Loading BioT5 models...")
        self.tokenizer = AutoTokenizer.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.logger.info("Model loading completed.")
        
    def sample_transition(self, node: MolStringNode) -> MolStringNode:
        parent_selfies = node.string
        prompt = self.prompt_prefix + " " + self.target_objective + self.prompt_postfix + " - Input: <bom>" + parent_selfies + "<eom> Output: "
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=512, do_sample=True)
        output_selfies = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
        
        return MolStringNode(string=output_selfies, lang=node.lang, parent=node)