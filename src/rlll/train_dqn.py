import gym as gym

from rlll.agents import DQNAgent
from rlll.nn import DQN
from rlll.utils import ExperienceReplayBuffer

lr = 0.001            #Velocidad de aprendizaje
MEMORY_SIZE = 100000  #Máxima capacidad del buffer
MAX_EPISODES = 5000   #Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
EPSILON = 1           #Valor inicial de epsilon
EPSILON_DECAY = .99   #Decaimiento de epsilon
GAMMA = 0.99          #Valor gamma de la ecuación de Bellman
BATCH_SIZE = 32       #Conjunto a coger del buffer para la red neuronal
BURN_IN = 1000        #Número de episodios iniciales usados para rellenar el buffer antes de entrenar
DNN_UPD = 1           #Frecuencia de actualización de la red neuronal
DNN_SYNC = 2500       #Frecuencia de sincronización de pesos entre la red neuronal y la red objetivo


env = gym.make('LunarLander-v2', render_mode='rgb_array')
buffer = ExperienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
dqn = DQN(env, learning_rate=lr)
agent = DQNAgent(env, dqn, buffer, EPSILON, EPSILON_DECAY, BATCH_SIZE)
agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES,
              batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC)
