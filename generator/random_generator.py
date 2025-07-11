from generator import Generator
from node import Node
from transition import Transition

class RandomGenerator(Generator):
    def __init__(self, root: Node, transition: Transition, max_length=None, **kwargs):
        self.root = root
        self.max_length = max_length or transition.max_length()
        super().__init__(transition=transition, **kwargs)
        
    # implement
    def _generate_impl(self):
        result = self.transition.rollout(self.root)
        self.get_objective_values_and_reward(result)