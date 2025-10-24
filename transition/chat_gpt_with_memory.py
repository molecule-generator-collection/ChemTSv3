import logging
from openai import OpenAI
from node import SMILESStringNode
from filter import Filter
from transition import LLMTransition

class ChatGPTTransitionWithMemory(LLMTransition):
    """Keeps conversation"""
    def __init__(self, prompt: str, initial_prompt: str=None, model: str="gpt-4o-mini", api_key: str=None, api_key_path: str=None, n_samples=1, filters: list[Filter]=None, logger: logging.Logger=None):
        if api_key is None and api_key_path is None:
            raise ValueError("Specify either 'api_key' or 'api_key_path'.")
        elif api_key is not None and api_key_path is not None:
            raise ValueError("Specify one of 'api_key' or 'api_key_path', not both.")
        elif api_key_path is not None:
            with open(api_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        self.client = OpenAI(api_key=api_key)
        
        super().__init__(prompt=prompt, n_samples=n_samples, filters=filters, logger=logger)
        
        self.response_id = None
        self.model = model
        self.sum_input_tokens = 0
        self.sum_output_tokens = 0

        if initial_prompt is not None:
            self.logger.debug(f"Prompt: '{initial_prompt}'")
            resp = self.client.responses.create(model=self.model, input=initial_prompt)
            self.response_id = resp.id
            self.sum_input_tokens += resp.usage.input_tokens
            self.sum_output_tokens += resp.usage.output_tokens
            self.logger.debug(f"Response: '{resp.output_text.strip()}'")

        self.prompt_queue = []
        
    # implement
    def observe(self, node: SMILESStringNode, objective_values: list[float], reward: float, is_filtered: bool):
        if not is_filtered:
            smiles = node.string
            text = f"The reward of molecule {smiles} was: {reward:.3f}."
            self.logger.debug(f"Prompt prefix: '{text}'")
            self.prompt_queue.append(text)
        super().observe(node, objective_values, reward, is_filtered)
        
    # implement
    def receive_response(self, prompt):
        for text in self.prompt_queue:
            prompt = text + "\n" + prompt
        self.prompt_queue = []
        
        resp = self.client.responses.create(model=self.model, input=prompt, previous_response_id=self.response_id)
        self.response_id = resp.id
        self.sum_input_tokens += resp.usage.input_tokens
        self.sum_output_tokens += resp.usage.output_tokens
        return resp.output_text.strip().replace("`", "")
        
    def analyze(self):
        self.logger.info(f"Total input tokens: {self.sum_input_tokens}")
        self.logger.info(f"Total output tokens: {self.sum_output_tokens}")
        super().analyze()