import gym as gym
import logging
from pathlib import Path
import click
import yaml

from rlll.agents import DQNAgent
from rlll.nn import DQN
from rlll.utils import ExperienceReplayBuffer


def train_dqn(model_output_path: str):
    params = yaml.safe_load(open("params.yaml"))["train"]
    lr = params['lr']                   # Velocidad de aprendizaje
    mem_size = params['mem_size']       # Máxima capacidad del buffer
    max_epi = params['max_epi']         # Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
    epsilon = params['epsilon']         # Valor inicial de epsilon
    epsilon_decay = params['epsilon']   # Decaimiento de epsilon
    gamma = params['gamma']             # Valor gamma de la ecuación de Bellman
    batch_size = params['batch_size']   # Conjunto a coger del buffer para la red neuronal
    burn_in = params['burn_in']         # Número de episodios iniciales usados para rellenar el buffer antes de entrenar
    dnn_update = params['dnn_update']   # Frecuencia de actualización de la red neuronal
    dnn_sync = params['dnn_sync']       # Frecuencia de sincronización de pesos entre la red neuronal y la red objetivo

    env = gym.make('LunarLander-v2')
    buffer = ExperienceReplayBuffer(memory_size=mem_size, burn_in=burn_in)
    dqn = DQN(env, learning_rate=lr)
    agent = DQNAgent(env, dqn, buffer, epsilon, epsilon_decay, batch_size)
    agent.train(gamma=gamma, max_episodes=max_epi,
                batch_size=batch_size, dnn_update_frequency=dnn_update, dnn_sync_frequency=dnn_sync)

    agent.save(model_output_path)


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Runs data processing scripts to turn processed data from (../split) into
        split data ready to be processed (saved in ../agg).
    """
    logger = logging.getLogger(__name__)
    logger.info('making train and test data set from processed data')

    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)

    train_dqn(output_path / 'model.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()