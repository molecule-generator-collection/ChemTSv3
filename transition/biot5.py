import logging
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from node import SELFIESStringNode
from transition import BlackBoxTransition

class BioT5Transition(BlackBoxTransition):
    def __init__(self, prompt: str, n_samples=2, logger: logging.Logger=None):
        self.prompt = prompt
        super().__init__(n_samples=n_samples, logger=logger)
        
        self.logger.info("Loading BioT5 models...")
        self.tokenizer = AutoTokenizer.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.logger.info("Model loading completed.")
        
    def sample_transition(self, node: SELFIESStringNode) -> SELFIESStringNode:
        parent_selfies = node.string
        prompt = self.prompt.replace("###SELFIES###", parent_selfies)
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=512, do_sample=True)
        output_selfies = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
        
        return SELFIESStringNode(string=output_selfies, parent=node)
    
class BioT5PlusTransition(BlackBoxTransition):
    def __init__(self, prompt: str, n_samples=2, logger: logging.Logger=None):
        self.prompt = prompt
        super().__init__(n_samples=n_samples, logger=logger)
        
        self.logger.info("Loading BioT5+ models...")
        self.tokenizer = AutoTokenizer.from_pretrained("QizhiPei/biot5-plus-base-mol-instructions-molecule")
        self.model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-plus-base-mol-instructions-molecule')
        self.logger.info("Model loading completed.")
        
    def sample_transition(self, node: SELFIESStringNode) -> SELFIESStringNode:
        parent_string = node.string
        prompt = self.prompt.replace("###SELFIES###", parent_string)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=512, do_sample=True)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=False).replace(" ", "")
        output = re.sub(r"<.*?>", "", output)
        
        return SELFIESStringNode(string=output, parent=node)