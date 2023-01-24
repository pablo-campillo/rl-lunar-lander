import numpy as np
import torch
import torch.nn as tnn


class DQN(tnn.Module):
    def __init__(self, env, input_size, hidden_size=16, learning_rate=1e-3, device='cpu'):
        super(DQN, self).__init__()
        self.env = env
        self.device = device
        self.n_inputs = input_size
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        ### Construcción de la red neuronal
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, self.hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.n_outputs, bias=True))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        ### Se ofrece la opción de trabajar con CUDA
        if self.device == 'cuda':
            self.model.cuda()

    ### Método e-greedy
    def get_action(self, state, epsilon=0.05):
        if self.env.np_random.random() < epsilon:
            action = self.env.np_random.choice(self.actions)  # acción aleatoria
        else:
            qvals = self.get_qvals(state)  # acción a partir del cálculo del valor de Q para esa acción
            action = torch.max(qvals, dim=-1)[1].item()
        return action

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.model(state_t)
