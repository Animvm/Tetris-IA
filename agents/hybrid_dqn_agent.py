import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from agents.dqn_agent import DQNAgent

# agente hibrido que combina imitacion y aprendizaje por refuerzo
class HybridDQNAgent(DQNAgent):
    def __init__(self, env, expert_data_path=None,
                 imitation_weight=1.0, **kwargs):
        super().__init__(env, **kwargs)

        self.imitation_weight = imitation_weight
        self.imitation_weight_initial = imitation_weight
        self.expert_memory = deque(maxlen=100000)

        if expert_data_path and os.path.exists(expert_data_path):
            self.load_expert_data(expert_data_path)

    # carga demostraciones de un experto
    def load_expert_data(self, path):
        print(f"Cargando datos expertos desde: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        for transition in data:
            self.expert_memory.append((
                transition['state'],
                transition['action'],
                transition['reward'],
                transition['next_state'],
                transition['done']
            ))

        print(f"Cargadas {len(self.expert_memory)} transiciones expertas")

    # entrena con loss hibrido imitacion y RL
    def train_step_hybrid(self):
        if len(self.memory) < self.batch_size // 2:
            return 0.0, 0.0, 0.0

        # dividir batch entre datos expertos y propios
        expert_batch_size = self.batch_size // 2
        agent_batch_size = self.batch_size - expert_batch_size

        expert_batch = []
        agent_batch = []

        if len(self.expert_memory) >= expert_batch_size:
            expert_batch = random.sample(list(self.expert_memory), expert_batch_size)

        if len(self.memory) >= agent_batch_size:
            agent_batch = random.sample(self.memory, agent_batch_size)

        batch = expert_batch + agent_batch
        if len(batch) < self.batch_size:
            return 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # loss de aprendizaje por refuerzo
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        td_loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # loss de imitacion del experto
        if len(expert_batch) > 0:
            expert_states = states[:len(expert_batch)]
            expert_actions = actions[:len(expert_batch)]

            q_values = self.policy_net(expert_states)
            imitation_loss = nn.CrossEntropyLoss()(q_values, expert_actions)
        else:
            imitation_loss = torch.tensor(0.0).to(self.device)

        # combinar losses
        total_loss = td_loss + self.imitation_weight * imitation_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # reducir peso de imitacion gradualmente
        self.imitation_weight *= 0.9999

        return total_loss.item(), td_loss.item(), imitation_loss.item()

    def get_imitation_weight_progress(self):
        if self.imitation_weight_initial == 0:
            return 1.0
        return 1.0 - (self.imitation_weight / self.imitation_weight_initial)
