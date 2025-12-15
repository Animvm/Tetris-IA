import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from agents.dqn_agent import DQNAgent

class HybridDQNAgent(DQNAgent):
    """
    Agente DQN hibrido que combina aprendizaje por refuerzo con imitacion.

    Aprende de dos fuentes:
    1. Demostraciones expertas de MCTS (behavioral cloning)
    2. Experiencia propia (Q-learning estandar)

    Loss combinado: L_total = L_TD + lambda * L_imitacion
    """

    def __init__(self, env, expert_data_path=None,
                 imitation_weight=1.0, **kwargs):
        """
        Args:
            env: ambiente de Tetris
            expert_data_path: ruta al archivo pickle con demos expertas
            imitation_weight: peso inicial del loss de imitacion (se reduce con el tiempo)
            **kwargs: argumentos para DQNAgent base
        """
        super().__init__(env, **kwargs)

        self.imitation_weight = imitation_weight
        self.imitation_weight_initial = imitation_weight
        self.expert_memory = deque(maxlen=100000)

        # Cargar datos expertos si se proporcionan
        if expert_data_path and os.path.exists(expert_data_path):
            self.load_expert_data(expert_data_path)

    def load_expert_data(self, path):
        """
        Carga dataset de demostraciones expertas desde archivo pickle.
        """
        print(f"Cargando datos expertos desde: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Convertir formato del generador al formato del buffer
        for transition in data:
            self.expert_memory.append((
                transition['state'],
                transition['action'],
                transition['reward'],
                transition['next_state'],
                transition['done']
            ))

        print(f"Cargadas {len(self.expert_memory)} transiciones expertas")

    def train_step_hybrid(self):
        """
        Paso de entrenamiento hibrido que combina:
        - TD loss: aprendizaje por refuerzo estandar
        - Imitation loss: imitacion de experto

        Estrategia:
        - Samplear 50% de buffer experto
        - Samplear 50% de experiencia propia
        - Calcular loss combinado
        - Annealing del peso de imitacion
        """
        if len(self.memory) < self.batch_size // 2:
            return 0.0, 0.0, 0.0

        # Determinar tamanos de batch
        expert_batch_size = self.batch_size // 2
        agent_batch_size = self.batch_size - expert_batch_size

        expert_batch = []
        agent_batch = []

        # Samplear de buffer experto si hay suficientes datos
        if len(self.expert_memory) >= expert_batch_size:
            expert_batch = random.sample(list(self.expert_memory), expert_batch_size)

        # Samplear de experiencia propia
        if len(self.memory) >= agent_batch_size:
            agent_batch = random.sample(self.memory, agent_batch_size)

        # Combinar batches
        batch = expert_batch + agent_batch
        if len(batch) < self.batch_size:
            return 0.0, 0.0, 0.0

        # Preparar tensores
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-learning loss (TD loss)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # DQN estandar
                next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        td_loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Imitation loss (solo para muestras expertas)
        if len(expert_batch) > 0:
            expert_states = states[:len(expert_batch)]
            expert_actions = actions[:len(expert_batch)]

            # Cross-entropy loss para imitacion
            q_values = self.policy_net(expert_states)
            imitation_loss = nn.CrossEntropyLoss()(q_values, expert_actions)
        else:
            imitation_loss = torch.tensor(0.0).to(self.device)

        # Loss combinado
        total_loss = td_loss + self.imitation_weight * imitation_loss

        # Optimizacion
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Actualizar target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Annealing: reducir peso de imitacion gradualmente
        # Permite transicion de pure imitation a pure RL
        self.imitation_weight *= 0.9999

        return total_loss.item(), td_loss.item(), imitation_loss.item()

    def get_imitation_weight_progress(self):
        """
        Retorna progreso del annealing de imitacion (0-1).
        0 = peso inicial, 1 = completamente reducido.
        """
        if self.imitation_weight_initial == 0:
            return 1.0
        return 1.0 - (self.imitation_weight / self.imitation_weight_initial)
