from abc import ABC, abstractmethod
from datetime import datetime
import logging
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from generator import Generator
from node import Node
from transition import WeightedTransition
from utils import camel2snake

class RandomGenerator(Generator):
    def __init__(self, root: Node, transition: WeightedTransition, max_length=None, rollout_conf = None, **kwargs):
        self.root = root
        self.transition = transition
        self.max_length = max_length or transition.max_length()
        self.rollout_count = 0
        self.rollout_conf = rollout_conf or {}
        super().__init__(**kwargs)
        
    # implement
    def _generate_impl(self):
        result = self.transition.rollout(self.root, **self.rollout_conf)
        self.grab_objective_values_and_reward(result)
        self.rollout_count += 1