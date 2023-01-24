import math

import numpy as np
from collections import namedtuple, deque


def normalize(state):
    return state / np.array([1.5, 1.5, 5, 5, math.pi, 5, 1, 1])


class LunarLanderState:
    @staticmethod
    def print_state(state):
        print(f"position: ({state[0]}, {state[1]})")
        print(f"velocity: ({state[2]}, {state[3]})")
        print(f"angle: {state[4]}")
        print(f"angular speed: {state[5]}")
        print(f"leg contact: ({state[6]}, {state[7]})")


class StackDataManager:
    def __init__(self, size: int):
        self.size = size
        self.stacked_data = None

    def add(self, state):
        #dt = 1/50.0
        #state[2] = state[0] * -state[2] * dt
        #state[3] = state[1] * -state[3] * dt
        #state[5] = state[4] * -state[5] * dt
        state = normalize(state)
        # angle_targ = state[0] * 0.5 + state[2] * 1.0
        # v1 = np.array([state[0], state[1]])
        # v2 = np.array([0, 1])
        # unit_vector_1 = v1 / np.linalg.norm(v1)
        # unit_vector_2 = v2 / np.linalg.norm(v2)
        # dot_product = np.dot(unit_vector_1, unit_vector_2)
        # angle = np.arccos(dot_product)
        # state[4] = angle - state[4]
        # return state
        if self.stacked_data is None:
            self.stacked_data = np.tile(state, self.size)
        else:
            self.stacked_data = np.roll(self.stacked_data, len(state))
            self.stacked_data[:len(state)] = state
        return self.stacked_data

    # def add(self, state):
    #     angle_targ = state[0] * 0.5 + state[2] * 1.0
    #     state[4] = angle_targ - state[4]
    #     if self.stacked_data is None:
    #         self.stacked_data = np.concatenate((state, np.zeros(8*(self.size-1)))) # np.tile(state, self.size)
    #     # else:
    #     #     self.stacked_data = np.roll(self.stacked_data, len(state))
    #     #     self.stacked_data[:len(state)] = state
    #     self.stacked_data = self._preprocess_stack(self.stacked_data, state)
    #     return self.stacked_data

    @staticmethod
    def _preprocess_stack(stack, state):
        a = np.reshape(stack, (-1, 8))
        a[0, :] -= state
        r = np.roll(a, 1, axis=0)
        r[0, :] = state
        return r.reshape(-1)


class ExperienceReplayBuffer:
    def __init__(self, env, memory_size=5000, burn_in=1000):
        self.env = env
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = namedtuple('Buffer',
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = self.env.np_random.choice(len(self.replay_memory), batch_size, replace=False)
        # Use el operador asterisco para desempaquetar deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(self.buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in
