import math

import gym
import os

from rlll.agents import RandAgent
from rlll.runners import StatsListener, GifListener, EnvRunManager

os.environ['SDL_VIDEODRIVER']='dummy'
import pygame
pygame.display.set_mode((640,480))
import pdb
import matplotlib.pyplot as plt


def play_random_once():
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = RandAgent(env.np_random)
    stats = StatsListener()
    gif = GifListener()
    env_run_manager = EnvRunManager(env, agent, seed=43, listeners=[stats, gif])
    with env_run_manager as erm:
        terminated = False
        while not terminated:
            terminated, _ = erm()

    stats.print()
    gif.save('gifs/random_agent_lunar-lander.gif')
    # imageio.mimwrite(os.path.join('.', 'random_agent_lunar-lander.gif'), frames, fps=60)

    plt.imshow(gif.frames[-1])
    # print(f"Initial state:")
    # print_state(frames[0])
    # print(f"Final state:")
    # print_state(frames[-1])
    # print(f"Number of steps: {len(hist_obs)}")
    # print(f"Total reward: {total_reward}")


if __name__ == '__main__':
    play_random_once()
