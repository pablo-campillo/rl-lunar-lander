# Created by Pablo Campillo at 14/1/23
import math
from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def step(self, obs: np.array) -> int:
        return math.trunc(self.env.np_random.random() * 4)


class RandAgent(Agent):

    def step(self, obs: np.array) -> int:
        return math.trunc(self.env.np_random.random() * 4)
