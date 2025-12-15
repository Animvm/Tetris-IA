import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import pandas as pd
from datetime import datetime
from envs.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from agents.mcts_agent import MCTSAgent
from agents.hybrid_dqn_agent import HybridDQNAgent
from agents.expert_generator import MCTSExpertGenerator

class ComparativeEvaluator:
    """
    Framework para comparacion rigurosa de agentes Tetris.
    Entrena y evalua DQN, MCTS y Hibrido bajo condiciones identicas.
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.results_dir = f"results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        print("="*70)
        print("EVALUACION COMPARATIVA - TETRIS IA")
        print("="*70)
        print(f"Resultados: {self.results_dir}")
        print(f"Seed: {self.seed}")
        print("="*70)

    def train_all_agents(self, episodes=1000, mcts_episodes=100):
        """
        Entrena los 3 agentes con las mismas condiciones.

        Args:
            episodes: episodios de entrenamiento/evaluacion
            mcts_episodes: episodios a evaluar para MCTS (no entrena)

        Returns:
            results: diccionario con metricas de cada agente
        """
        results = {}

        # 1. DQN Mejorado
        print("\n" + "="*70)
        print("AGENTE 1: DQN Mejorado")
        print("="*70)
        env = TetrisEnv(seed=self.seed, use_action_masking=True)
        dqn_agent = DQNAgent(
            env,
            lr=0.00025,
            buffer_size=50000,
            batch_size=64,
            target_update=1000,
            epsilon_decay=0.9995,
            epsilon_min=0.05,
            use_double_dqn=True
        )
        results['dqn'] = self.train_agent(dqn_agent, env, episodes, "DQN")
        dqn_agent.save(os.path.join(self.results_dir, "dqn_final.pth"))

        # 2. MCTS Puro (evaluacion directa, sin entrenamiento)
        print("\n" + "="*70)
        print("AGENTE 2: MCTS Puro")
        print("="*70)
        env = TetrisEnv(seed=self.seed, use_action_masking=True)
        mcts_agent = MCTSAgent(env, num_simulations=100, max_profundidad=10)
        results['mcts'] = self.evaluate_agent(mcts_agent, env, mcts_episodes, "MCTS")

        # 3. Hibrido MCTS-DQN
        print("\n" + "="*70)
        print("AGENTE 3: Hibrido MCTS-DQN")
        print("="*70)

        # Generar datos expertos
        print("Generando datos expertos para hibrido...")
        env_expert = TetrisEnv(seed=self.seed, use_action_masking=True)
        expert_gen = MCTSExpertGenerator(env_expert, num_simulations=200,
                                        save_dir=os.path.join(self.results_dir, "expert_data"))
        dataset, expert_path = expert_gen.generate_dataset(num_episodes=100, min_score=50)

        env = TetrisEnv(seed=self.seed, use_action_masking=True)
        hybrid_agent = HybridDQNAgent(
            env,
            expert_data_path=expert_path,
            imitation_weight=1.0,
            lr=0.0001,
            buffer_size=50000,
            batch_size=64,
            target_update=1000,
            use_double_dqn=True
        )

        # Pre-entrenamiento
        print("\nPre-entrenamiento hibrido (500 pasos)...")
        for ep in range(500):
            for _ in range(10):
                if len(hybrid_agent.expert_memory) >= hybrid_agent.batch_size:
                    hybrid_agent.train_step_hybrid()

        # Fine-tuning
        results['hybrid'] = self.train_agent_hybrid(hybrid_agent, env, episodes, "Hybrid")
        hybrid_agent.save(os.path.join(self.results_dir, "hybrid_final.pth"))

        # Guardar todos los resultados
        self.save_results(results)
        self.generate_report(results)

        return results

    def train_agent(self, agent, env, episodes, name):
        """Entrena un agente DQN estandar y registra metricas."""
        metrics = {
            'scores': [],
            'lines': [],
            'pieces': [],
            'rewards': [],
            'computation_time': [],
            'episode_lengths': []
        }

        print(f"Entrenando {name} por {episodes} episodios...")

        for ep in range(episodes):
            start_time = time.time()

            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                valid_actions = env.get_valid_actions() if env.use_action_masking else None
                action = agent.select_action(obs, training=True, valid_actions=valid_actions)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(obs, action, reward, next_obs, done)
                agent.train_step()

                total_reward += reward
                steps += 1
                obs = next_obs

            episode_time = time.time() - start_time

            metrics['scores'].append(info['score'])
            metrics['lines'].append(info['lines'])
            metrics['pieces'].append(steps)
            metrics['rewards'].append(total_reward)
            metrics['computation_time'].append(episode_time)
            metrics['episode_lengths'].append(steps)

            if (ep + 1) % 100 == 0:
                avg_score = np.mean(metrics['scores'][-100:])
                avg_lines = np.mean(metrics['lines'][-100:])
                print(f"{name} Ep {ep+1:4d}/{episodes}: "
                      f"Avg Score={avg_score:6.1f}, "
                      f"Avg Lines={avg_lines:4.1f}")

        return metrics

    def train_agent_hybrid(self, agent, env, episodes, name):
        """Entrena agente hibrido con train_step_hybrid."""
        metrics = {
            'scores': [],
            'lines': [],
            'pieces': [],
            'rewards': [],
            'computation_time': [],
            'episode_lengths': []
        }

        print(f"Entrenando {name} por {episodes} episodios...")

        for ep in range(episodes):
            start_time = time.time()

            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                valid_actions = env.get_valid_actions() if env.use_action_masking else None
                action = agent.select_action(obs, training=True, valid_actions=valid_actions)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(obs, action, reward, next_obs, done)
                agent.train_step_hybrid()

                total_reward += reward
                steps += 1
                obs = next_obs

            episode_time = time.time() - start_time

            metrics['scores'].append(info['score'])
            metrics['lines'].append(info['lines'])
            metrics['pieces'].append(steps)
            metrics['rewards'].append(total_reward)
            metrics['computation_time'].append(episode_time)
            metrics['episode_lengths'].append(steps)

            if (ep + 1) % 100 == 0:
                avg_score = np.mean(metrics['scores'][-100:])
                avg_lines = np.mean(metrics['lines'][-100:])
                print(f"{name} Ep {ep+1:4d}/{episodes}: "
                      f"Avg Score={avg_score:6.1f}, "
                      f"Avg Lines={avg_lines:4.1f}")

        return metrics

    def evaluate_agent(self, agent, env, episodes, name):
        """Evalua agente sin entrenamiento (para MCTS)."""
        metrics = {
            'scores': [],
            'lines': [],
            'pieces': [],
            'rewards': [],
            'computation_time': [],
            'episode_lengths': []
        }

        print(f"Evaluando {name} por {episodes} episodios...")

        for ep in range(episodes):
            start_time = time.time()

            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = agent.seleccionar_accion(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                steps += 1
                obs = next_obs

            episode_time = time.time() - start_time

            metrics['scores'].append(info['score'])
            metrics['lines'].append(info['lines'])
            metrics['pieces'].append(steps)
            metrics['rewards'].append(total_reward)
            metrics['computation_time'].append(episode_time)
            metrics['episode_lengths'].append(steps)

            if (ep + 1) % 20 == 0:
                avg_score = np.mean(metrics['scores'][-20:])
                avg_lines = np.mean(metrics['lines'][-20:])
                print(f"{name} Ep {ep+1:3d}/{episodes}: "
                      f"Avg Score={avg_score:6.1f}, "
                      f"Avg Lines={avg_lines:4.1f}")

        return metrics

    def save_results(self, results):
        """Guarda metricas de cada agente en CSV."""
        print("\nGuardando resultados...")

        for name, metrics in results.items():
            df = pd.DataFrame(metrics)
            csv_path = os.path.join(self.results_dir, f"{name}_metrics.csv")
            df.to_csv(csv_path, index=False)
            print(f"  {name}: {csv_path}")

    def generate_report(self, results):
        """Genera reporte de comparacion en texto."""
        report_path = os.path.join(self.results_dir, "comparison_report.txt")

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE COMPARACION - TETRIS IA\n")
            f.write("="*70 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Seed: {self.seed}\n\n")

            f.write("RESUMEN DE PERFORMANCE\n")
            f.write("-"*70 + "\n\n")

            # Tabla comparativa
            f.write(f"{'Agente':<15} {'Score Avg':<12} {'Lines Avg':<12} {'Max Score':<12} {'Tiempo/Ep (s)':<15}\n")
            f.write("-"*70 + "\n")

            for name, metrics in results.items():
                avg_score = np.mean(metrics['scores'][-100:]) if len(metrics['scores']) >= 100 else np.mean(metrics['scores'])
                avg_lines = np.mean(metrics['lines'][-100:]) if len(metrics['lines']) >= 100 else np.mean(metrics['lines'])
                max_score = np.max(metrics['scores'])
                avg_time = np.mean(metrics['computation_time'])

                f.write(f"{name.upper():<15} {avg_score:<12.1f} {avg_lines:<12.1f} {max_score:<12.1f} {avg_time:<15.2f}\n")

            f.write("\n" + "="*70 + "\n")

        print(f"\nReporte generado: {report_path}")

if __name__ == "__main__":
    evaluator = ComparativeEvaluator(seed=42)
    results = evaluator.train_all_agents(episodes=1000, mcts_episodes=100)

    print("\n" + "="*70)
    print("EVALUACION COMPARATIVA COMPLETADA")
    print("="*70)
