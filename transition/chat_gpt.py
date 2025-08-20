import logging
import os
from openai import OpenAI
from node import SMILESStringNode
from transition import BlackBoxTransition

class ChatGPTTransition(BlackBoxTransition):
    def __init__(self, prompt: str, model: str="gpt-4o-mini", api_key: str=None, api_key_path: str=None, n_samples=2, logger: logging.Logger=None):
        if api_key is None and api_key_path is None:
            raise ValueError("Specify either 'api_key' or 'api_key_path'.")
        elif api_key is not None and api_key_path is not None:
            raise ValueError("Specify one of 'api_key' or 'api_key_path', not both.")
        elif api_key_path is not None:
            with open(api_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        self.api_key = api_key
        
        if not isinstance(prompt, list):
            prompt = [prompt]
        self.prompt = prompt
        
        self.n = 0
        self.sum_deltas_unfiltered = [0] * len(self.prompt)
        self.sum_deltas_including_filtered = [0] * len(self.prompt)
        self.n_filtered = [0] * len(self.prompt)
        
        self.model = model
        self.sum_input_tokens = 0
        self.sum_output_tokens = 0
        
        super().__init__(n_samples=n_samples, logger=logger)
        
    # implement
    def sample_transition(self, node: SMILESStringNode) -> SMILESStringNode:
        parent_smiles = node.string
        
        results = []
        for i, p in enumerate(self.prompt):
            prompt = p.replace("###SMILES###", parent_smiles)
            self.logger.debug(f"Prompt: '{prompt}'")
            
            client = OpenAI(api_key=self.api_key)
            resp = client.responses.create(model=self.model, input=prompt)
            
            self.sum_input_tokens += resp.usage.input_tokens
            self.sum_output_tokens += resp.usage.output_tokens
            output_smiles = resp.output_text.strip()
            self.logger.debug(f"Response: '{output_smiles}', input_tokens: {resp.usage.input_tokens}, output_tokens: {resp.usage.output_tokens}")
            results.append(SMILESStringNode(string=output_smiles, parent=node, last_action=i))
        
        self.n += 1
        return results
    
    # implement
    def observe(self, node: SMILESStringNode, objective_values: list[float], reward: float, filtered: bool):
        action = node.last_action
        if node.parent.reward is None:
            return
        dif = reward - node.parent.reward
        if not filtered:
            self.sum_deltas_unfiltered[action] += dif
            self.sum_deltas_including_filtered[action] += dif
        else:
            self.sum_deltas_including_filtered[action] += dif
            self.n_filtered[action] += 1
    
    def analyze(self):
        self.logger.info(f"Total conversations: {self.n} * {len(self.prompt)} = {self.n * len(self.prompt)}")
        self.logger.info(f"Total input tokens: {self.sum_input_tokens}")
        self.logger.info(f"Total output tokens: {self.sum_output_tokens}")
        for i in range(len(self.prompt)):
            self.logger.info(f"------------------------- Prompt {i} -------------------------")
            self.logger.info(f"Average delta (unfiltered): {self.sum_deltas_unfiltered[i] / self.n}")
            self.logger.info(f"Average delta (with filtered): {self.sum_deltas_including_filtered[i] / self.n}")
            self.logger.info(f"Number of filtered output: {self.n_filtered[i]}")

class LongChatGPTTransition(BlackBoxTransition):
    """Keeps conversation"""
    def __init__(self, prompt: str, initial_prompt: str=None, model: str="gpt-4o-mini", api_key: str=None, api_key_path: str=None, n_samples=2, logger: logging.Logger=None):
        if api_key is None and api_key_path is None:
            raise ValueError("Specify either 'api_key' or 'api_key_path'.")
        elif api_key is not None and api_key_path is not None:
            raise ValueError("Specify one of 'api_key' or 'api_key_path', not both.")
        elif api_key_path is not None:
            with open(api_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        self.client = OpenAI(api_key=api_key)
        
        super().__init__(n_samples=n_samples, logger=logger)
        
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
            self.logger.debug(f"Response: '{resp.output_text.strip()}', input_tokens: {resp.usage.input_tokens}, output_tokens: {resp.usage.output_tokens}")
        
        if not isinstance(prompt, list):
            prompt = [prompt]
        self.prompt = prompt
        
        self.observation_record = []
        
    # implement
    def observe(self, node: SMILESStringNode, objective_values: list[float], reward: float, filtered: bool):
        if not filtered:
            smiles = node.string
            text = f"The reward of molecule with SMILES {smiles} was: {reward:.3f}."
            self.observation_record.append(text)
        
    # implement
    def sample_transition(self, node: SMILESStringNode) -> SMILESStringNode:
        parent_smiles = node.string
        
        results = []
        for i, p in enumerate(self.prompt):
            prompt = p.replace("###SMILES###", parent_smiles)
            if i == 0: # Not actually needed
                for text in self.observation_record:
                    prompt = text + "\n" + prompt
                self.observation_record = []
            self.logger.debug(f"Prompt: '{prompt}'")
            
            resp = self.client.responses.create(model=self.model, input=prompt, previous_response_id=self.response_id)
            self.response_id = resp.id
            self.sum_input_tokens += resp.usage.input_tokens
            self.sum_output_tokens += resp.usage.output_tokens
            output_smiles = resp.output_text.strip()
            self.logger.debug(f"Response: '{output_smiles}', input_tokens: {resp.usage.input_tokens}, output_tokens: {resp.usage.output_tokens}")
            results.append(SMILESStringNode(string=output_smiles, parent=node))
        
        return results
    
    def analyze(self):
        self.logger.info(f"Sum input tokens: {self.sum_input_tokens}")
        self.logger.info(f"Sum output tokens: {self.sum_output_tokens}")