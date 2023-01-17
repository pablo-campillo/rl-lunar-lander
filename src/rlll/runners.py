# Created by Pablo Campillo at 14/1/23
from abc import abstractmethod, ABC

import imageio


class EnvListener(ABC):
    @abstractmethod
    def new_step(self, env, state, reward, terminated, truncated, info):
        pass


class EnvRunManager:
    """ Class to run an environment and an agent provided in the constructor.
    """
    def __init__(self, env, agent, seed=43, listeners=[]):
        self.env = env
        self.agent = agent
        self.seed = seed
        self.current_state = None
        self.terminated = False
        self.listeners = listeners

    def __enter__(self):
        self.terminated = False
        self.current_state, info = self.env.reset(seed=self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()

    def __call__(self, *args, **kwargs):
        action = self.agent.step(self.current_state)
        self.current_state, reward, terminated, truncated, info = self.env.step(action)
        self._notify_listeners(self.current_state, reward, terminated, truncated, info)
        return terminated, truncated

            #if terminated or truncated:
            #    observation, info = env.reset()

    def _notify_listeners(self, current_state, reward, terminated, truncated, info):
        for l in self.listeners:
            l.new_step(self.env, current_state, reward, terminated, truncated, info)


class StatsListener(EnvListener):
    def __init__(self):
        self.total_rewared = 0
        self.num_steps = 0

    def new_step(self, env, state, reward, terminated, truncated, info):
        self.total_rewared += reward
        self.num_steps += 1


class GifListener(EnvListener):
    def __init__(self):
        self.frames = []

    def new_step(self, env, state, reward, terminated, truncated, info):
        frame = self.env.render()
        self.frames.append(frame)

    def save(self, file_path, fps=60):
        imageio.mimwrite(file_path, self.frames, fps=fps)
