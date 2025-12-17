import os
import pickle
import numpy as np
from agents.mcts_agent import MCTSAgent

# genera demostraciones expertas usando MCTS
class MCTSExpertGenerator:
    def __init__(self, env, num_simulations=200, save_dir="data/expert_demos"):
        self.env = env
        self.mcts = MCTSAgent(env, num_simulaciones=num_simulations, max_profundidad=10)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # genera un episodio completo
    def generate_episode(self):
        trajectory = []
        obs, _ = self.env.reset()
        done = False

        while not done:
            action = self.mcts.seleccionar_accion(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            trajectory.append({
                'state': obs.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_obs.copy(),
                'done': done,
                'lines': info.get('lines', 0),
                'score': info.get('score', 0)
            })

            obs = next_obs

        return trajectory

    # genera dataset de demostraciones filtrando por calidad
    def generate_dataset(self, num_episodes=100, min_score=50):
        dataset = []
        episode_count = 0

        print(f"Generando {num_episodes} episodios expertos con MCTS...")
        print(f"Simulaciones por accion: {self.mcts.num_simulaciones}")
        print(f"Filtro de calidad: score >= {min_score} o lineas >= 5")
        print("="*60)

        for ep in range(num_episodes):
            trajectory = self.generate_episode()
            score = trajectory[-1]['score']
            lines = trajectory[-1]['lines']
            num_steps = len(trajectory)

            # guardar solo episodios de buena calidad
            if score >= min_score or lines >= 5:
                dataset.extend(trajectory)
                episode_count += 1
                print(f"Ep {ep+1:3d}: Score={score:6.1f}, Lines={lines:2d}, Steps={num_steps:3d} - Guardado")
            else:
                print(f"Ep {ep+1:3d}: Score={score:6.1f}, Lines={lines:2d}, Steps={num_steps:3d} - Descartado")

        # guardar dataset en archivo
        save_path = os.path.join(self.save_dir, f"expert_data_{episode_count}eps.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)

        print("\n" + "="*60)
        print(f"Dataset generado:")
        print(f"  Episodios guardados: {episode_count}/{num_episodes}")
        print(f"  Total de transiciones: {len(dataset)}")
        print(f"  Archivo: {save_path}")

        return dataset, save_path
