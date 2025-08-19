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
        
        self.model = model
        super().__init__(n_samples=n_samples, logger=logger)
        
    # implement
    def sample_transition(self, node: SMILESStringNode) -> SMILESStringNode:
        parent_smiles = node.string
        
        results = []
        for p in self.prompt:
            prompt = p.replace("###SMILES###", parent_smiles)
            
            client = OpenAI(api_key=self.api_key)
            resp = client.responses.create(model=self.model, input=prompt)
            output_smiles = resp.output_text
            results.append(SMILESStringNode(string=output_smiles, parent=node))
        
        return results

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
        
        self.response_id = None
        self.model = model

        if initial_prompt is not None:
            resp = self.client.responses.create(model=self.model, input=initial_prompt)
            self.response_id = resp.id    
        
        if not isinstance(prompt, list):
            prompt = [prompt]
        self.prompt = prompt
        
        self.observation_record = []
        
        super().__init__(n_samples=n_samples, logger=logger)
        
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
            if i == 0:
                for text in self.observation_record:
                    prompt = text + "\n" + prompt
                self.observation_record = []
            self.logger.debug(prompt)
            
            resp = self.client.responses.create(model=self.model, input=prompt, previous_response_id=self.response_id)
            self.response_id = resp.id
            output_smiles = resp.output_text.strip()
            results.append(SMILESStringNode(string=output_smiles, parent=node))
        
        return results