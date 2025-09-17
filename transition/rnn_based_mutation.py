import logging
from language import Language
from node import MolSentenceNode, MolStringNode, FASTAStringNode
from transition import RNNTransition, TemplateTransition
from utils import find_lang_file

class RNNBasedMutation(TemplateTransition):
    """RNN should return the sentence of the same length with a reasonable chance."""
    def __init__(self, n_samples: int=1, n_tries: int=5, model_dir: str=None, device: str=None, rnn_top_p: float=1.0, rnn_temperature: float=1.0, rnn_sharpness: float=1.0, filters=None, logger: logging.Logger=None):
        self.n_samples = n_samples
        self.n_tries = n_tries
        lang_path = find_lang_file(model_dir)
        self.lang = Language.load(lang_path)
        self.rnn = RNNTransition(lang=self.lang, model_dir=model_dir, device=device, top_p=rnn_top_p, temperature=rnn_temperature, sharpness=rnn_sharpness, logger=logger)
        self.empty_node = MolSentenceNode.node_from_key(key="", lang=self.lang, device=device)
        super().__init__(filters=filters, logger=logger)
    
    def _next_nodes_impl(self, node: MolStringNode | FASTAStringNode):
        candidates = []
        for _ in range(self.n_samples):
            base_sentence = node.key()
            rnd_sentence = ""
            count = -1
            while len(rnd_sentence) != len(base_sentence):
                count += 1
                if count >= self.n_tries:
                    self.logger.warning(f"The number of key length unmatches exceeded the threshold ({self.n_tries}) from: {base_sentence}")
                    return []
                rnd_sentence = self.rnn.rollout(self.empty_node).key()
            for j in range(len(base_sentence)):
                if base_sentence[j] != rnd_sentence[j]:
                    new_sentence = base_sentence[:j] + rnd_sentence[j] + base_sentence[j+1:]
                    if not new_sentence in candidates:
                        candidates.append(new_sentence)
        
        if isinstance(node, FASTAStringNode):
            return [FASTAStringNode.node_from_key(key=c, parent=node, last_prob=1/len(candidates)) for c in candidates]
        else:
            return [MolStringNode.node_from_key(key=c, lang=self.lang, parent=node, last_prob=1/len(candidates)) for c in candidates]