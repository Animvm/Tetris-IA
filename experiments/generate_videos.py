import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pygame
from envs.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from agents.hybrid_dqn_agent import HybridDQNAgent
import imageio

def record_gameplay(agent, env, output_path, num_episodes=3, max_steps_per_episode=1000):
    """
    Graba el juego del agente y genera un video.

    Args:
        agent: Agente entrenado (DQNAgent o HybridDQNAgent)
        env: Entorno de Tetris
        output_path: Ruta donde se guardara el video
        num_episodes: Numero de episodios a grabar
        max_steps_per_episode: Maximo de pasos por episodio
    """
    print(f"\nGenerando video: {output_path}")
    print(f"Episodios: {num_episodes}")
    print("-" * 70)

    frames = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        episode_score = 0
        episode_lines = 0

        while not done and steps < max_steps_per_episode:
            # Obtener acciones validas
            valid_actions = env.get_valid_actions() if env.use_action_masking else None

            # Seleccionar accion (sin exploracion)
            action = agent.select_action(obs, training=False, valid_actions=valid_actions)

            # Ejecutar accion
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Renderizar y capturar frame
            env.render()
            frame = env.get_rgb_array()
            if frame is not None:
                # Convertir RGB array a formato para video
                frames.append(frame)

            steps += 1
            episode_score = info.get('score', episode_score)
            episode_lines = info.get('lines', episode_lines)

        print(f"  Episodio {ep + 1}/{num_episodes}: "
              f"Score={episode_score:.0f}, "
              f"Lineas={episode_lines}, "
              f"Pasos={steps}")

    # Guardar video
    if frames:
        print(f"\nGuardando video con {len(frames)} frames...")
        imageio.mimsave(output_path, frames, fps=10)
        print(f"Video guardado exitosamente: {output_path}")
    else:
        print("ERROR: No se capturaron frames")

    return len(frames)


def generate_dqn_parallel_video():
    """Genera video del modelo DQN Parallel"""
    print("=" * 70)
    print("GENERANDO VIDEO: DQN PARALLEL")
    print("=" * 70)

    # Configurar entorno con renderizado
    env = TetrisEnv(use_action_masking=True, render_mode="rgb_array")

    # Cargar agente
    model_path = "results/dqn_parallel_20251217_011426/dqn_parallel_final.pth"
    agent = DQNAgent(env)
    agent.load(model_path)
    agent.epsilon = 0.0  # Sin exploracion para mejor visualizacion

    print(f"\nModelo cargado: {model_path}")
    print(f"Device: {agent.device}")

    # Generar video
    output_path = "videos/dqn_parallel_gameplay.mp4"
    os.makedirs("videos", exist_ok=True)

    record_gameplay(agent, env, output_path, num_episodes=3, max_steps_per_episode=1000)

    env.close()
    return output_path


def generate_hybrid_video():
    """Genera video del modelo Hybrid"""
    print("\n" + "=" * 70)
    print("GENERANDO VIDEO: HYBRID (MCTS-DQN)")
    print("=" * 70)

    # Configurar entorno con renderizado
    env = TetrisEnv(use_action_masking=True, render_mode="rgb_array")

    # Cargar agente hibrido
    model_path = "results/hybrid_20251217_013657/hybrid_model_final.pth"

    # Crear agente hibrido sin datos expertos (solo para inferencia)
    agent = HybridDQNAgent(
        env,
        expert_data_path=None,
        imitation_weight=0.0,
        lr=0.0001
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # Sin exploracion para mejor visualizacion

    print(f"\nModelo cargado: {model_path}")
    print(f"Device: {agent.device}")

    # Generar video
    output_path = "videos/hybrid_gameplay.mp4"
    os.makedirs("videos", exist_ok=True)

    record_gameplay(agent, env, output_path, num_episodes=3, max_steps_per_episode=1000)

    env.close()
    return output_path


def main():
    """Genera ambos videos"""
    print("\nGENERADOR DE VIDEOS DE MODELOS FINALES")
    print("=" * 70)

    # Verificar que pygame este disponible
    try:
        import pygame
        pygame.init()
        print("Pygame inicializado correctamente")
    except Exception as e:
        print(f"ERROR: Problema con pygame: {e}")
        return

    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Usando CPU")

    try:
        # Generar video DQN Parallel
        video1 = generate_dqn_parallel_video()

        # Generar video Hybrid
        video2 = generate_hybrid_video()

        # Resumen
        print("\n" + "=" * 70)
        print("VIDEOS GENERADOS EXITOSAMENTE")
        print("=" * 70)
        print(f"1. DQN Parallel: {video1}")
        print(f"2. Hybrid MCTS-DQN: {video2}")
        print("\nPuedes reproducir los videos con cualquier reproductor de video.")

    except Exception as e:
        print(f"\nERROR durante la generacion de videos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
