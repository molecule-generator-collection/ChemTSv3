import logging
from openai import OpenAI
from node import SMILESStringNode
from transition import BlackBoxTransition

class ChatGPTTransition(BlackBoxTransition):
    def __init__(self, api_key: str, prompt: str, model: str="gpt-4o-mini", n_samples=2, logger: logging.Logger=None):
        self.api_key = api_key
        self.prompt = prompt
        self.model = model
        super().__init__(n_samples=n_samples, logger=logger)
        
    # implement
    def sample_transition(self, node: SMILESStringNode) -> SMILESStringNode:
        parent_smiles = node.string
        prompt = self.prompt.replace("###SMILES###", parent_smiles)
        
        client = OpenAI(api_key=self.api_key)
        resp = client.responses.create(model=self.model, input=prompt)
        output_smiles = resp.output_text
        
        return SMILESStringNode(string=output_smiles, parent=node)