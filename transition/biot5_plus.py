import logging
import re
from transformers import AutoTokenizer, T5ForConditionalGeneration
from node import MolStringNode
from transition import BlackBoxTransition

class BioT5PlusTransition(BlackBoxTransition):
    def __init__(self, target_objective: str, prompt_prefix: str=None, prompt_postfix: str=None, n_samples=2, logger: logging.Logger=None):
        self.target_objective = target_objective
        self.prompt_prefix = prompt_prefix or "The molecule is obtained by performing a single molecular editing (such as amide bond formation, Suzuki–Miyaura coupling, and SNAr reactions) on: "
        self.prompt_postfix = prompt_postfix or ". The output molecule must be connected (must not have '.' in selfies)."
        super().__init__(logger=logger)
        
        self.logger.info("Loading BioT5+ models...")
        self.tokenizer = AutoTokenizer.from_pretrained("QizhiPei/biot5-plus-base-mol-instructions-molecule")
        self.model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-plus-base-mol-instructions-molecule')
        self.logger.info("Model loading completed.")
        
    def sample_transition(self, node: MolStringNode) -> MolStringNode:
        parent_string = node.string
        prompt = self.prompt_prefix + parent_string + " to " + self.target_objective + self.prompt_postfix

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=512, do_sample=True)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=False).replace(" ", "")
        output = re.sub(r"<.*?>", "", output)
        
        return MolStringNode(string=output, lang=node.lang, parent=node)