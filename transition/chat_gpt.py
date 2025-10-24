import logging
from openai import OpenAI
from filter import Filter
from transition import LLMTransition

class ChatGPTTransition(LLMTransition):
    def __init__(self, prompt: str, model: str="gpt-4o-mini", api_key: str=None, api_key_path: str=None, n_samples=1, filters: list[Filter]=None, logger: logging.Logger=None):
        if api_key is None and api_key_path is None:
            raise ValueError("Specify either 'api_key' or 'api_key_path'.")
        elif api_key is not None and api_key_path is not None:
            raise ValueError("Specify one of 'api_key' or 'api_key_path', not both.")
        elif api_key_path is not None:
            with open(api_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        self.api_key = api_key
        
        self.model = model
        self.sum_input_tokens = 0
        self.sum_output_tokens = 0
        
        super().__init__(prompt=prompt, n_samples=n_samples, filters=filters, logger=logger)
        
    # implement    
    def receive_response(self, prompt):
        client = OpenAI(api_key=self.api_key)
        resp = client.responses.create(model=self.model, input=prompt)
        
        self.sum_input_tokens += resp.usage.input_tokens
        self.sum_output_tokens += resp.usage.output_tokens
        return resp.output_text.strip().replace("`", "")
    
    def analyze(self):
        self.logger.info(f"Total input tokens: {self.sum_input_tokens}")
        self.logger.info(f"Total output tokens: {self.sum_output_tokens}")
        super().analyze()