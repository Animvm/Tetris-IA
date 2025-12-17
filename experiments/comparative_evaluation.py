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
from utils.parallel_env import ParallelEnv
from utils.statistical_analysis import statistical_comparison, print_statistical_report
from utils.comparison_plots import plot_comparison

def make_env():
    """Funcion para crear entorno (debe estar en top-level para pickle en Windows)."""
    return TetrisEnv(use_action_masking=True)

class ComparativeEvaluator:
    """
    Framework para comparacion rigurosa de agentes Tetris.
    Entrena y evalua DQN, MCTS y Hibrido bajo condiciones identicas.
    Todos los entrenamientos usan paralelizacion para acelerar el proceso.
    """

    def __init__(self, seed=42, num_parallel_envs=8):
        self.seed = seed
        self.num_parallel_envs = num_parallel_envs
        self.results_dir = f"results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        print("="*70)
        print("EVALUACION COMPARATIVA - TETRIS IA")
        print("="*70)
        print(f"Resultados: {self.results_dir}")
        print(f"Seed: {self.seed}")
        print(f"Entornos paralelos: {self.num_parallel_envs}")
        print("="*70)

    def train_all_agents(self, episodes=2000, mcts_episodes=100):
        """
        Entrena los 3 agentes con las mismas condiciones usando paralelizacion.

        Args:
            episodes: episodios de entrenamiento/evaluacion para DQN y Hibrido
            mcts_episodes: episodios a evaluar para MCTS (no entrena)

        Returns:
            results: diccionario con metricas de cada agente
        """
        results = {}

        # 1. DQN Mejorado con Paralelizacion
        print("\n" + "="*70)
        print("AGENTE 1: DQN Mejorado (Paralelizado)")
        print("="*70)
        results['dqn'], dqn_agent = self.train_dqn_parallel(episodes)
        dqn_agent.save(os.path.join(self.results_dir, "dqn_final.pth"))

        # 2. MCTS Puro (evaluacion directa, sin entrenamiento)
        print("\n" + "="*70)
        print("AGENTE 2: MCTS Puro")
        print("="*70)
        env = make_env()
        mcts_agent = MCTSAgent(env, num_simulaciones=100, max_profundidad=10)
        results['mcts'] = self.evaluate_agent(mcts_agent, env, mcts_episodes, "MCTS")

        # 3. Hibrido MCTS-DQN con Paralelizacion
        print("\n" + "="*70)
        print("AGENTE 3: Hibrido MCTS-DQN (Paralelizado)")
        print("="*70)
        results['hybrid'], hybrid_agent = self.train_hybrid_parallel(episodes)
        hybrid_agent.save(os.path.join(self.results_dir, "hybrid_final.pth"))

        # Guardar todos los resultados
        self.save_results(results)
        self.generate_report(results)

        # Analisis estadistico
        self.statistical_analysis(results)

        # Generar graficos comparativos
        print("\nGenerando graficos comparativos...")
        plot_comparison(results, self.results_dir)

        return results

    def train_dqn_parallel(self, episodes):
        """Entrena DQN usando multiples entornos en paralelo."""
        print(f"Entrenando DQN por {episodes} episodios (paralelizado)...")

        # Crear entornos paralelos
        parallel_envs = ParallelEnv(make_env, num_envs=self.num_parallel_envs)

        # Crear agente
        single_env = make_env()
        agent = DQNAgent(
            single_env,
            lr=0.00025,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9995,
            buffer_size=100000,
            batch_size=128,
            target_update=5000,
            use_double_dqn=True
        )

        # Metricas
        metrics = {
            'scores': [],
            'lines': [],
            'pieces': [],
            'rewards': [],
            'computation_time': [],
            'episode_lengths': []
        }

        # Estado inicial
        obs_batch, _ = parallel_envs.reset()
        episode_count = 0
        episode_rewards = [0.0] * self.num_parallel_envs
        episode_steps = [0] * self.num_parallel_envs
        episode_start_times = [time.time()] * self.num_parallel_envs

        while episode_count < episodes:
            # Obtener acciones validas para cada entorno
            valid_actions_batch = parallel_envs.get_valid_actions()

            # Seleccionar acciones usando batch inference
            actions = agent.select_actions_batch(
                obs_batch,
                training=True,
                valid_actions_list=valid_actions_batch
            )

            # Ejecutar step en paralelo
            next_obs_batch, rewards, terminateds, truncateds, infos = parallel_envs.step(actions)

            # Procesar cada entorno
            for i in range(self.num_parallel_envs):
                agent.store_transition(
                    obs_batch[i],
                    actions[i],
                    rewards[i],
                    next_obs_batch[i],
                    terminateds[i]
                )

                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1

                # Si el episodio termino
                if terminateds[i] or truncateds[i]:
                    episode_time = time.time() - episode_start_times[i]

                    metrics['scores'].append(infos[i]['score'])
                    metrics['lines'].append(infos[i]['lines'])
                    metrics['pieces'].append(episode_steps[i])
                    metrics['rewards'].append(episode_rewards[i])
                    metrics['computation_time'].append(episode_time)
                    metrics['episode_lengths'].append(episode_steps[i])

                    episode_count += 1

                    if episode_count % 50 == 0:
                        avg_score = np.mean(metrics['scores'][-50:])
                        avg_lines = np.mean(metrics['lines'][-50:])
                        print(f"DQN Ep {episode_count:4d}/{episodes}: "
                              f"Avg Score={avg_score:6.1f}, "
                              f"Avg Lines={avg_lines:4.1f}, "
                              f"ε={agent.epsilon:.3f}")

                        # Visualizar 1 episodio cada 50 para ver progreso
                        print(f"  -> Mostrando episodio de demostracion...")
                        self.visualize_episode(agent, episode_count)

                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    episode_start_times[i] = time.time()

                    if episode_count >= episodes:
                        break

            # Entrenar agente (4 steps balanceado CPU/GPU)
            for _ in range(4):
                if len(agent.memory) >= agent.batch_size:
                    agent.train_step()

            obs_batch = next_obs_batch

        parallel_envs.close()
        return metrics, agent

    def train_hybrid_parallel(self, episodes):
        """Entrena agente Hibrido usando multiples entornos en paralelo."""
        # Generar datos expertos primero
        print("Generando datos expertos para hibrido...")
        env_expert = make_env()
        expert_gen = MCTSExpertGenerator(
            env_expert,
            num_simulations=200,
            save_dir=os.path.join(self.results_dir, "expert_data")
        )
        dataset, expert_path = expert_gen.generate_dataset(num_episodes=100, min_score=50)

        print(f"\nEntrenando Hibrido por {episodes} episodios (paralelizado)...")

        # Crear entornos paralelos
        parallel_envs = ParallelEnv(make_env, num_envs=self.num_parallel_envs)

        # Crear agente hibrido
        single_env = make_env()
        agent = HybridDQNAgent(
            single_env,
            expert_data_path=expert_path,
            imitation_weight=1.0,
            lr=0.0001,
            epsilon_min=0.1,
            epsilon_decay=0.9995,
            buffer_size=100000,
            batch_size=128,
            target_update=5000,
            use_double_dqn=True
        )

        # Pre-entrenamiento con imitacion pura
        print("\nPre-entrenamiento hibrido (500 episodios de imitacion)...")
        for ep in range(500):
            for _ in range(10):
                if len(agent.expert_memory) >= agent.batch_size:
                    agent.train_step_hybrid()
            if (ep + 1) % 100 == 0:
                print(f"  Pre-training: {ep+1}/500 episodios, imitation_weight={agent.imitation_weight:.4f}")

        # Fine-tuning con entornos paralelos
        print("\nFine-tuning hibrido con entornos paralelos...")

        # Metricas
        metrics = {
            'scores': [],
            'lines': [],
            'pieces': [],
            'rewards': [],
            'computation_time': [],
            'episode_lengths': []
        }

        # Estado inicial
        obs_batch, _ = parallel_envs.reset()
        episode_count = 0
        episode_rewards = [0.0] * self.num_parallel_envs
        episode_steps = [0] * self.num_parallel_envs
        episode_start_times = [time.time()] * self.num_parallel_envs

        while episode_count < episodes:
            valid_actions_batch = parallel_envs.get_valid_actions()

            actions = agent.select_actions_batch(
                obs_batch,
                training=True,
                valid_actions_list=valid_actions_batch
            )

            next_obs_batch, rewards, terminateds, truncateds, infos = parallel_envs.step(actions)

            for i in range(self.num_parallel_envs):
                agent.store_transition(
                    obs_batch[i],
                    actions[i],
                    rewards[i],
                    next_obs_batch[i],
                    terminateds[i]
                )

                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1

                if terminateds[i] or truncateds[i]:
                    episode_time = time.time() - episode_start_times[i]

                    metrics['scores'].append(infos[i]['score'])
                    metrics['lines'].append(infos[i]['lines'])
                    metrics['pieces'].append(episode_steps[i])
                    metrics['rewards'].append(episode_rewards[i])
                    metrics['computation_time'].append(episode_time)
                    metrics['episode_lengths'].append(episode_steps[i])

                    episode_count += 1

                    if episode_count % 50 == 0:
                        avg_score = np.mean(metrics['scores'][-50:])
                        avg_lines = np.mean(metrics['lines'][-50:])
                        print(f"Hybrid Ep {episode_count:4d}/{episodes}: "
                              f"Avg Score={avg_score:6.1f}, "
                              f"Avg Lines={avg_lines:4.1f}, "
                              f"imit_weight={agent.imitation_weight:.4f}")

                        # Visualizar 1 episodio cada 50 para ver progreso
                        print(f"  -> Mostrando episodio de demostracion...")
                        self.visualize_episode(agent, episode_count)

                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    episode_start_times[i] = time.time()

                    if episode_count >= episodes:
                        break

            # Entrenar con loss hibrido (4 steps balanceado CPU/GPU)
            for _ in range(4):
                if len(agent.memory) >= agent.batch_size:
                    agent.train_step_hybrid()

            obs_batch = next_obs_batch

        parallel_envs.close()
        return metrics, agent

    def evaluate_agent(self, agent, env, episodes, name):
        """Evalua un agente ya entrenado (MCTS)."""
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

    def visualize_episode(self, agent, episode_num):
        """
        Ejecuta y muestra visualmente 1 episodio con el agente actual.
        Util para ver como juega el agente durante el entrenamiento.
        """
        # Crear entorno con rendering
        vis_env = TetrisEnv(use_action_masking=True, render_mode="human")

        obs, _ = vis_env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"     [Visualizacion Ep {episode_num}] Presiona la ventana para continuar, cierra para saltar")

        while not done and steps < 200:  # Max 200 piezas para no hacer muy largo
            # Obtener acciones validas
            valid_actions = vis_env.get_valid_actions()

            # Seleccionar accion (sin exploracion, modo greedy)
            action = agent.select_action(obs, training=False, valid_actions=valid_actions)

            # Ejecutar accion
            next_obs, reward, terminated, truncated, info = vis_env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            obs = next_obs

            # Renderizar
            vis_env.render()
            time.sleep(0.3)  # Pausa para ver las piezas caer

        vis_env.close()

        print(f"     [Resultado] Score: {info['score']:.1f}, Lineas: {info['lines']}, Piezas: {steps}")

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
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Entornos paralelos: {self.num_parallel_envs}\n\n")

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

    def statistical_analysis(self, results):
        """Realiza analisis estadistico comparativo."""
        print("\nGenerando analisis estadistico...")
        report_path = os.path.join(self.results_dir, "statistical_report.txt")

        # Comparar DQN vs MCTS
        dqn_vs_mcts = statistical_comparison(
            results['dqn']['scores'],
            results['mcts']['scores'],
            "DQN",
            "MCTS"
        )

        # Comparar DQN vs Hybrid
        dqn_vs_hybrid = statistical_comparison(
            results['dqn']['scores'],
            results['hybrid']['scores'],
            "DQN",
            "Hybrid"
        )

        # Comparar MCTS vs Hybrid
        mcts_vs_hybrid = statistical_comparison(
            results['mcts']['scores'],
            results['hybrid']['scores'],
            "MCTS",
            "Hybrid"
        )

        # Imprimir y guardar reporte
        with open(report_path, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print("="*70)
            print("ANALISIS ESTADISTICO COMPARATIVO")
            print("="*70)
            print()

            print_statistical_report(dqn_vs_mcts)
            print("\n" + "-"*70 + "\n")
            print_statistical_report(dqn_vs_hybrid)
            print("\n" + "-"*70 + "\n")
            print_statistical_report(mcts_vs_hybrid)

            sys.stdout = original_stdout

        print(f"Analisis estadistico guardado: {report_path}")

if __name__ == "__main__":
    evaluator = ComparativeEvaluator(seed=42, num_parallel_envs=8)

    # Entrenar con mas episodios gracias a la paralelizacion
    results = evaluator.train_all_agents(episodes=2000, mcts_episodes=100)

    print("\n" + "="*70)
    print("EVALUACION COMPARATIVA COMPLETADA")
    print("="*70)
    print(f"\nResultados guardados en: {evaluator.results_dir}")
    print("\nArchivos generados:")
    print("  - dqn_final.pth, hybrid_final.pth (modelos entrenados)")
    print("  - dqn_metrics.csv, mcts_metrics.csv, hybrid_metrics.csv (datos)")
    print("  - comparison_report.txt (resumen)")
    print("  - statistical_report.txt (analisis estadistico)")
    print("  - comparison_overview.png (grafico comparativo)")
