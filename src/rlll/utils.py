import numpy as np
from collections import namedtuple, deque


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
        if self.stacked_data is None:
            self.stacked_data = np.stack([state] * self.size)
        else:
            self.stacked_data = np.stack([state] + [self.stacked_data[i] for i in range(self.size - 1)])
        return self.stacked_data


class ExperienceReplayBuffer:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = namedtuple('Buffer',
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use el operador asterisco para desempaquetar deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in
