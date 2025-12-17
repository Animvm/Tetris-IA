import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# red neuronal para estimar valores Q
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # capas convolucionales para procesar el tablero
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        conv_out_size = input_shape[0] * input_shape[1] * 128

        # capas densas para generar valores Q
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_actions)
        )

    # procesa el estado y retorna valores Q
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# agente que aprende a jugar usando DQN
class DQNAgent:
    def __init__(self, env, lr=0.00025, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.9995, buffer_size=100000,
                 batch_size=128, target_update=5000, use_double_dqn=True):
        self.env = env
        self.n_actions = env.action_space.n
        self.input_shape = env.observation_space.shape

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_double_dqn = use_double_dqn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        self.steps = 0

    # selecciona accion usando epsilon-greedy
    def select_action(self, state, training=True, valid_actions=None):
        # exploracion aleatoria
        if training and random.random() < self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.n_actions)

        # explotacion usando red neuronal
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

            # aplicar mascara de acciones validas
            if valid_actions is not None:
                q_values_masked = q_values.clone()
                mask = torch.ones(self.n_actions, dtype=torch.bool)
                mask[valid_actions] = False
                q_values_masked[0, mask] = -float('inf')
                return q_values_masked.argmax().item()

            return q_values.argmax().item()

    # selecciona acciones en lote para entornos paralelos
    def select_actions_batch(self, states, training=True, valid_actions_list=None):
        batch_size = len(states)
        actions = []

        # generar mascara para exploracion
        if training:
            random_mask = np.random.random(batch_size) < self.epsilon
        else:
            random_mask = np.zeros(batch_size, dtype=bool)

        # acciones aleatorias para exploracion
        random_actions = []
        for i in range(batch_size):
            if random_mask[i]:
                if valid_actions_list and valid_actions_list[i]:
                    random_actions.append(random.choice(valid_actions_list[i]))
                else:
                    random_actions.append(random.randrange(self.n_actions))

        # acciones greedy usando la red neuronal
        if not random_mask.all():
            greedy_indices = np.where(~random_mask)[0]
            greedy_states = states[greedy_indices]

            with torch.no_grad():
                state_tensor = torch.FloatTensor(greedy_states).to(self.device)
                q_values = self.policy_net(state_tensor)

                # aplicar mascaras de acciones validas
                if valid_actions_list:
                    for i, idx in enumerate(greedy_indices):
                        if valid_actions_list[idx]:
                            mask = torch.ones(self.n_actions, dtype=torch.bool)
                            mask[valid_actions_list[idx]] = False
                            q_values[i, mask] = -float('inf')

                greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        random_idx = 0
        greedy_idx = 0
        for i in range(batch_size):
            if random_mask[i]:
                actions.append(random_actions[random_idx])
                random_idx += 1
            else:
                actions.append(int(greedy_actions[greedy_idx]))
                greedy_idx += 1

        return actions

    # guarda experiencia en el buffer de replay
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # entrena la red con un batch de experiencias
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        # muestrear batch del buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convertir a tensores
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # entrenamiento con precision mixta si hay GPU
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                # calcular Q-values actuales
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

                # calcular targets usando double DQN o DQN clasico
                with torch.no_grad():
                    if self.use_double_dqn:
                        next_actions = self.policy_net(next_states).argmax(1)
                        next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                    else:
                        next_q = self.target_net(next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * self.gamma * next_q

                loss = nn.MSELoss()(current_q.squeeze(), target_q)

            # actualizar pesos con gradient clipping
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # entrenamiento sin precision mixta
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

            with torch.no_grad():
                if self.use_double_dqn:
                    next_actions = self.policy_net(next_states).argmax(1)
                    next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                else:
                    next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q

            loss = nn.MSELoss()(current_q.squeeze(), target_q)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        self.steps += 1
        # actualizar red objetivo periodicamente
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # decaer epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    # guarda el modelo entrenado
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    # carga un modelo previamente entrenado
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
