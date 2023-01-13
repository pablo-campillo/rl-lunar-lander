import math

import gym
import os

from rlll.utils import print_state

os.environ['SDL_VIDEODRIVER']='dummy'
import pygame
pygame.display.set_mode((640,480))
import pdb
import imageio
import matplotlib.pyplot as plt


class RandAgent:
    def __init__(self, np_random):
        self.np_random = np_random

    def step(self, obs):
        return math.trunc(self.np_random.random() * 4)


def play_random_once():
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    observation, info = env.reset(seed=42)

    agent = RandAgent(env.np_random)
    total_reward = 0
    terminated = False
    hist_obs = []
    frames = []
    while not terminated:
        action = agent.step(observation)
        frame = env.render()
        frames.append(frame)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        hist_obs.append(observation)

        #if terminated or truncated:
        #    observation, info = env.reset()
    env.close()
    imageio.mimwrite(os.path.join('.', 'random_agent_lunar-lander.gif'), frames, fps=60)

    plt.imshow(frames[-1])
    print(f"Initial state:")
    print_state(frames[0])
    print(f"Final state:")
    print_state(frames[-1])
    print(f"Number of steps: {len(hist_obs)}")
    print(f"Total reward: {total_reward}")


if __name__ == '__main__':
    play_random_once()
