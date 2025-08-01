import heapq
from generator import Generator
from node import Node
from transition import Transition

class HeapQueueGenerator(Generator):
    def __init__(self, root: Node, transition: Transition, max_length=None, **kwargs):
        self.root = root
        self.max_length = max_length or transition.max_length()
        self.q = []
        self.checked_keys = set()
        heapq.heappush(self.q, (0, self.root.key(), self.root))
        super().__init__(transition=transition, **kwargs)
        
    # implement
    def _generate_impl(self):
        _, _, node = heapq.heappop(self.q)
        _, children, _ = zip(*self.transition.transitions(node))
        for child in children:
            key = child.key()
            if key in self.checked_keys:
                continue
            else:
                self.checked_keys.add(key)
                _, reward = self._eval(child)
                heapq.heappush(self.q, (-reward, key, child))
                
    def _eval(self, node: Node):
        if node.has_reward():
            objective_values, reward = self._get_objective_values_and_reward(node)
            node.reward = reward
        else:
            offspring = self.transition.rollout(node)
            objective_values, reward = self._get_objective_values_and_reward(offspring)
        return objective_values, reward