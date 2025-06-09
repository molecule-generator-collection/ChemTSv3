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
    def __init__(self, transition: WeightedTransition, max_length=None, **kwargs):
        self.root = None
        self.transition = transition
        self.max_length = max_length or transition.max_length()
        super().__init__(**kwargs)
        
    # implement
    def _generate_impl(self):
        pass
        