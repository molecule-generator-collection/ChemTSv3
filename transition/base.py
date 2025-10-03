from abc import ABC, abstractmethod
import logging
import random
from typing import Any
import numpy as np
from filter import Filter
from language import Language
from node import Node, SentenceNode

class Transition(ABC):
    def __init__(self, logger: logging.Logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    @abstractmethod
    def next_nodes(self, node: Node) -> list[Node]:
        """Return the list of the child nodes. If the node is terminal, an empty list [] should be returned."""
        pass

    def rollout(self, initial_node: Node) -> Node:
        """Sample an offspring with has_reward() = True"""
        if initial_node.is_terminal():
            return initial_node
        
        current_node = initial_node
        while True:
            transitions = self.transitions(current_node)
            if not transitions:
                current_node.mark_as_terminal()
                return current_node
            
            _, next_nodes, probs = zip(*transitions)
            next_node = random.choices(next_nodes, weights=probs, k=1)[0]
            
            if next_node.has_reward():
                return next_node
            
            current_node = next_node
            
    def transitions(self, node: Node) -> list[tuple[Any, Node, float]]:
        """Return the list of (action, node, probability) tuples."""
        transitions = []
        for node in self.next_nodes(node):
            action = node.last_action
            prob = node.last_prob
            transitions.append((action, node, prob))
        return transitions
    
    def max_length(self) -> int:
        """Returns the maximum number of transitions from a single node. Should be overridden if not infinite."""
        return 10**18
    
    def observe(self, node: Node, objective_values: list[float], reward: float, filtered: bool):
        """Transitions can update their internal state when observing the reward of the node. By default, this method does nothing."""
        
    def analyze(self):
        """This method is called within Generation.analyze(). By default, this method does nothing."""
    
class TemplateTransition(Transition):
    def __init__(self, filters: list[Filter]=None, top_p: float=None, logger: logging.Logger=None):
        if filters is None:
            filters = []
        self.filters = filters
        if top_p is not None and (top_p <= 0 or 1 < top_p):
            raise ValueError(f"Invalid top_p range: {top_p}")
        self.top_p = top_p
        self.filter_counts = [0] * len(filters)
        super().__init__(logger)
        
    @abstractmethod
    def _next_nodes_impl(self, node: Node) -> list[Node]:
        pass

    def next_nodes(self, node: Node):
        raw_nodes = self._next_nodes_impl(node)
        result = []
        
        # apply filters
        result = []
        for n in raw_nodes:
            passed = True
            for i, f in enumerate(self.filters):
                if not f.check(n):
                    self.filter_counts[i] += 1
                    passed = False
                    break
            if passed:
                result.append(n)
        if not result:
            return []
        
        # normalize after filters
        total = sum(c.last_prob for c in result)
        if total > 0:
            for c in result:
                c.last_prob /= total
        else:
            uniform = 1.0 / len(result)
            for c in result:
                c.last_prob = uniform

        # apply top_p
        if self.top_p is not None and len(result) > 0:
            probs = np.array([cand.last_prob for cand in result], dtype=float)
            probs = probs / probs.sum()

            sorted_idx = np.argsort(-probs)
            cumprobs = np.cumsum(probs[sorted_idx])

            keep_mask = cumprobs <= self.top_p
            if not keep_mask.any():
                keep_mask[0] = True # leave at least 1
            kept_idx = sorted_idx[keep_mask]

            result = [result[i] for i in kept_idx]

            # renormalize
            total = sum(c.last_prob for c in result)
            for c in result:
                c.last_prob /= total

        return result
    
    # override
    def analyze(self):
        if len(self.filters) != 0:
            self.logger.info(f"Filter counts (transition): {self.filter_counts}")
    
class BlackBoxTransition(TemplateTransition):
    def __init__(self, n_samples=2, filters: list[Filter]=None, logger: logging.Logger=None):
        self.n_samples = n_samples
        super().__init__(filters=filters, logger=logger)    

    @abstractmethod
    def sample_transition(self, node: Node) -> Node | list[Node] | None:
        """Sample child nodes."""
        pass
    
    # implement
    def _next_nodes_impl(self, node):
        children = []
        for i in range(self.n_samples):
            next_nodes = self.sample_transition(node)
            
            if next_nodes is None:
                next_nodes = []
            if not isinstance(next_nodes, list):
                next_nodes = [next_nodes]
                
            for child in next_nodes:
                if child.last_action is None: # Don't override the action label if it already exists
                    child.last_action = i
                children.append(child)
        for child in children:
            child.last_prob = 1 / len(children)
        return children
    
class AutoRegressiveTransition(Transition):
    def __init__(self, lang: Language, logger: logging.Logger=None):
        self.lang = lang
        super().__init__(logger)

    @abstractmethod
    def next_nodes(self, node: SentenceNode) -> list[Node]:
        pass

    # override
    def max_length(self) -> int:
        return 10**18
    
# class LLMTransition(BlackBoxTransition):
#     def __init__(self, prompt: str, n_samples=2, logger: logging.Logger=None):
#         if not isinstance(prompt, list):
#             prompt = [prompt]
#         self.prompt = prompt
        
#         self.n = 0
#         self.sum_deltas_unfiltered = [0] * len(self.prompt)
#         self.sum_deltas_including_filtered = [0] * len(self.prompt)
#         self.n_filtered = [0] * len(self.prompt)
        
#         super().__init__(n_samples=n_samples, logger=logger)
        
#     # implement
#     def sample_transition(self, node: SMILESStringNode) -> SMILESStringNode:
#         parent_smiles = node.string
        
#         results = []
#         for i, p in enumerate(self.prompt):
#             prompt = p.replace("###SMILES###", parent_smiles)
#             self.logger.debug(f"Prompt: '{prompt}'")
            
#             client = OpenAI(api_key=self.api_key)
#             resp = client.responses.create(model=self.model, input=prompt)
            
#             self.sum_input_tokens += resp.usage.input_tokens
#             self.sum_output_tokens += resp.usage.output_tokens
#             output_smiles = resp.output_text.strip()
#             self.logger.debug(f"Response: '{output_smiles}', input_tokens: {resp.usage.input_tokens}, output_tokens: {resp.usage.output_tokens}")
#             results.append(SMILESStringNode(string=output_smiles, parent=node, last_action=i))
        
#         self.n += 1
#         return results
    
#     # implement
#     def observe(self, node: SMILESStringNode, objective_values: list[float], reward: float, filtered: bool):
#         action = node.last_action
#         if node.parent.reward is None:
#             return
#         dif = reward - node.parent.reward
#         if not filtered:
#             self.sum_deltas_unfiltered[action] += dif
#             self.sum_deltas_including_filtered[action] += dif
#         else:
#             self.sum_deltas_including_filtered[action] += dif
#             self.n_filtered[action] += 1
    
#     def analyze(self):
#         self.logger.info(f"Total conversations: {self.n} * {len(self.prompt)} = {self.n * len(self.prompt)}")
#         self.logger.info(f"Total input tokens: {self.sum_input_tokens}")
#         self.logger.info(f"Total output tokens: {self.sum_output_tokens}")
#         for i in range(len(self.prompt)):
#             self.logger.info(f"------------------------- Prompt {i} -------------------------")
#             self.logger.info(f"Average delta (unfiltered): {self.sum_deltas_unfiltered[i] / self.n}")
#             self.logger.info(f"Average delta (with filtered): {self.sum_deltas_including_filtered[i] / self.n}")
#             self.logger.info(f"Number of filtered output: {self.n_filtered[i]}")