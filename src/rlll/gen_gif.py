import json

import gym as gym
import logging
from pathlib import Path
import click
import yaml

from rlll.agents import DQNAgent
from rlll.nn import DQN
from rlll.runners import StatsListener, GifListener, EnvRunManager
from rlll.utils import ExperienceReplayBuffer


def play_random_once(env, agent):
    stats = StatsListener()
    gif = GifListener()
    env_run_manager = EnvRunManager(env, agent, seed=43, listeners=[stats, gif])
    with env_run_manager as erm:
        terminated = False
        truncated = False
        while not terminated and not truncated:
            terminated, truncated = erm()

    print(f"num steps: {stats.num_steps}")
    gif.save('gifs/dqn_agent_lunar-lander.gif')
    # imageio.mimwrite(os.path.join('.', 'random_agent_lunar-lander.gif'), frames, fps=60)


def gen_gif():
    params = yaml.safe_load(open("params.yaml"))["train"]
    lr = params['lr']                           # Velocidad de aprendizaje
    mem_size = params['mem_size']               # Máxima capacidad del buffer
    epsilon = params['epsilon']                 # Valor inicial de epsilon
    epsilon_decay = params['epsilon_decay']     # Decaimiento de epsilon
    batch_size = params['batch_size']           # Conjunto a coger del buffer para la red neuronal
    burn_in = params['burn_in']                 # Número de episodios iniciales usados para rellenar el buffer antes de entrenar
    stack_size = params['stack_size']
    hidden_size = params['hidden_size']

    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    buffer = ExperienceReplayBuffer(memory_size=mem_size, burn_in=burn_in)
    dqn = DQN(env, 8*stack_size, hidden_size=hidden_size, learning_rate=lr, device='cuda')
    agent = DQNAgent(env, dqn, buffer, stack_size, epsilon, epsilon_decay, batch_size)
    agent.load('models/model.pkl')
    play_random_once(env, agent)


@click.command()
def main():
    logger = logging.getLogger(__name__)
    logger.info('making train and test data set from processed data')

    # output_path = Path(output_filepath)
    # output_path.mkdir(parents=True, exist_ok=True)

    gen_gif()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
