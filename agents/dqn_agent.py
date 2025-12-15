# agents/dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    """
    Red neuronal convolucional para aproximar Q-values.
    Arquitectura mejorada con batch normalization y dropout para mejor generalizacion.
    """
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
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

        # Tamaño de salida de convolucion
        conv_out_size = input_shape[0] * input_shape[1] * 128

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_actions)
        )
        
    def forward(self, x):
        # x shape: (batch, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    """
    Agente DQN con mejoras criticas:
    - Buffer mas grande (50k vs 10k original)
    - Target network actualizado menos frecuente (1000 pasos vs 10 original)
    - Epsilon decay mas lento para mejor exploracion
    - Double DQN para reducir overestimation
    """
    def __init__(self, env, lr=0.00025, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9995, buffer_size=50000,
                 batch_size=64, target_update=1000, use_double_dqn=True):
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
        
        # Networks
        self.policy_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        
        self.steps = 0
        
    def select_action(self, state, training=True, valid_actions=None):
        """
        Selecciona una accion usando epsilon-greedy.
        Si se provee valid_actions, solo considera acciones validas.
        """
        if training and random.random() < self.epsilon:
            # Exploracion: accion aleatoria
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.n_actions)

        # Explotacion: mejor accion segun Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

            # Aplicar mascara de acciones invalidas
            if valid_actions is not None:
                q_values_masked = q_values.clone()
                mask = torch.ones(self.n_actions, dtype=torch.bool)
                mask[valid_actions] = False
                q_values_masked[0, mask] = -float('inf')
                return q_values_masked.argmax().item()

            return q_values.argmax().item()

    def select_actions_batch(self, states, training=True, valid_actions_list=None):
        """
        Selecciona acciones para un batch de estados (optimizado para GPU).
        Util para entrenamiento paralelo con multiples entornos.

        Args:
            states: array de estados (num_envs, height, width)
            training: si True, aplica epsilon-greedy
            valid_actions_list: lista de listas con acciones validas por entorno

        Returns:
            lista de acciones seleccionadas
        """
        batch_size = len(states)
        actions = []

        # Determinar que acciones seran aleatorias (epsilon-greedy)
        if training:
            random_mask = np.random.random(batch_size) < self.epsilon
        else:
            random_mask = np.zeros(batch_size, dtype=bool)

        # Acciones aleatorias
        random_actions = []
        for i in range(batch_size):
            if random_mask[i]:
                if valid_actions_list and valid_actions_list[i]:
                    random_actions.append(random.choice(valid_actions_list[i]))
                else:
                    random_actions.append(random.randrange(self.n_actions))

        # Acciones greedy (batch inference en GPU)
        if not random_mask.all():
            greedy_indices = np.where(~random_mask)[0]
            greedy_states = states[greedy_indices]

            with torch.no_grad():
                state_tensor = torch.FloatTensor(greedy_states).to(self.device)
                q_values = self.policy_net(state_tensor)

                # Aplicar mascaras de acciones invalidas
                if valid_actions_list:
                    for i, idx in enumerate(greedy_indices):
                        if valid_actions_list[idx]:
                            mask = torch.ones(self.n_actions, dtype=torch.bool)
                            mask[valid_actions_list[idx]] = False
                            q_values[i, mask] = -float('inf')

                greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        # Combinar acciones aleatorias y greedy
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

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Valores Q actuales
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Valores Q objetivo (target)
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: seleccionar accion con policy net, evaluar con target net
                # Reduce overestimation bias
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # DQN estandar
                next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
