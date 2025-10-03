import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transition import LLMTransition

class BioT5Transition(LLMTransition):
    def __init__(self, prompt: str, n_samples=1, logger: logging.Logger=None):
        super().__init__(prompt=prompt, n_samples=n_samples, logger=logger)
        
        self.logger.info("Loading BioT5 models...")
        self.tokenizer = AutoTokenizer.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.logger.info("Model loading completed.")
        
    def receive_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=512, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")