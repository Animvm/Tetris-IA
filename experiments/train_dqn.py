import sys
sys.path.append('.')

# train_dqn.py
import time
import pygame
import numpy as np
import torch
from envs.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from utils.plotting import plot_metrics
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def pygame_delay(ms):
    """Small helper to keep the window responsive while waiting"""
    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < ms:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        pygame.time.delay(10)

def replay_episode(actions, delay_ms=100):
    """
    Reproducir un episodio usando las acciones guardadas
    
    Args:
        actions: lista de acciones a reproducir
        delay_ms: delay entre pasos (ms)
    """
    replay_env = TetrisEnv(render_mode="human", cell_size=28)
    
    try:
        obs, _ = replay_env.reset()
        replay_env.render()
        pygame_delay(1000)  # Pausa inicial
        
        for step, action in enumerate(actions):
            obs, reward, done, truncated, info = replay_env.step(action)
            replay_env.render()
            pygame_delay(delay_ms)
            
            if done:
                break
        
        # Pausa final para ver el resultado
        print(f"\n✓ Reproducción completada. Score final: {info.get('score', 0)}")
        pygame_delay(3000)
        
    except KeyboardInterrupt:
        print("\n⚠️  Reproducción interrumpida")
    finally:
        replay_env.close()

def run(episodes=500, visual_interval=50, delay_ms=50, save_model=True, replay_best=True,
        checkpoint_interval=100, use_tensorboard=True):
    """
    Entrenar DQN con visualización periódica y logging detallado

    Args:
        episodes: número total de episodios
        visual_interval: cada cuántos episodios mostrar visualización
        delay_ms: delay entre pasos en visualización (ms)
        save_model: si guardar el modelo entrenado
        replay_best: si reproducir el mejor episodio al finalizar
        checkpoint_interval: cada cuántos episodios guardar checkpoint
        use_tensorboard: si usar TensorBoard para logging
    """
    # Crear directorios con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Inicializar TensorBoard
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(f"{run_dir}/tensorboard")

    # Entorno y agente
    env = TetrisEnv()
    agent = DQNAgent(env, lr=0.0001, gamma=0.99, epsilon=1.0,
                     epsilon_decay=0.995, batch_size=32)

    # Guardar configuración del entrenamiento
    config = {
        'timestamp': timestamp,
        'episodes': episodes,
        'lr': 0.0001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': agent.epsilon_min,
        'batch_size': 32,
        'buffer_size': agent.memory.maxlen,
        'target_update': agent.target_update,
        'device': str(agent.device)
    }
    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

    # Entorno visual (se crea solo cuando se necesita)
    visual_env = None

    # Métricas detalladas
    rewards = []
    lines = []
    losses = []
    scores = []
    pieces_placed = []
    episode_lengths = []
    q_values_avg = []

    # Tracking del mejor episodio
    best_episode = {
        'episode': 0,
        'score': -float('inf'),
        'lines': 0,
        'actions': []
    }

    # Inicializar archivo de log CSV
    log_file = f"{run_dir}/training_log.csv"
    with open(log_file, 'w') as f:
        f.write("episode,reward,score,lines,pieces,steps,avg_loss,epsilon,q_value_avg,timestamp\n")

    print("="*70)
    print("ENTRENAMIENTO DQN - TETRIS IA")
    print("="*70)
    print(f"Device: {agent.device}")
    print(f"Episodios: {episodes}")
    print(f"Resultados en: {run_dir}")
    if use_tensorboard:
        print(f"TensorBoard: tensorboard --logdir={run_dir}/tensorboard")
    print("="*70 + "\n")
    
    start_time = time.time()

    try:
        for ep in range(episodes):
            # Determinar si este episodio se visualiza
            visualize = (ep + 1) % visual_interval == 0

            if visualize and visual_env is None:
                visual_env = TetrisEnv(render_mode="human", cell_size=24)

            # Usar entorno visual o normal
            current_env = visual_env if visualize else env

            obs, _ = current_env.reset()
            done = False
            total_r = 0
            total_lines = 0
            episode_loss = []
            episode_actions = []
            episode_q_values = []
            steps = 0

            if visualize:
                current_env.render()

            while not done:
                # Seleccionar acción y obtener Q-value
                action = agent.select_action(obs, training=True)

                # Calcular Q-value promedio para análisis
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                    q_vals = agent.policy_net(state_tensor)
                    episode_q_values.append(q_vals.mean().item())

                episode_actions.append(action)
                next_obs, reward, done, truncated, info = current_env.step(action)

                # Almacenar transición
                agent.store_transition(obs, action, reward, next_obs, done)

                # Entrenar
                loss = agent.train_step()
                if loss > 0:
                    episode_loss.append(loss)

                total_r += reward
                total_lines = info.get("lines", total_lines)
                obs = next_obs
                steps += 1

                if visualize:
                    current_env.render()
                    pygame_delay(delay_ms)

            # Registrar métricas
            episode_score = info.get('score', total_r)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            avg_q = np.mean(episode_q_values) if episode_q_values else 0

            rewards.append(total_r)
            lines.append(total_lines)
            losses.append(avg_loss)
            scores.append(episode_score)
            pieces_placed.append(steps)
            episode_lengths.append(steps)
            q_values_avg.append(avg_q)

            # Logging a CSV
            with open(log_file, 'a') as f:
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ep+1},{total_r},{episode_score},{total_lines},{steps},{steps},"
                       f"{avg_loss},{agent.epsilon},{avg_q},{timestamp_str}\n")

            # TensorBoard logging
            if writer:
                writer.add_scalar('Training/Reward', total_r, ep)
                writer.add_scalar('Training/Score', episode_score, ep)
                writer.add_scalar('Training/Lines', total_lines, ep)
                writer.add_scalar('Training/Loss', avg_loss, ep)
                writer.add_scalar('Training/Epsilon', agent.epsilon, ep)
                writer.add_scalar('Training/Steps', steps, ep)
                writer.add_scalar('Training/Avg_Q_Value', avg_q, ep)

                # Promedios móviles
                if len(rewards) >= 10:
                    writer.add_scalar('Training/Reward_MA10', np.mean(rewards[-10:]), ep)
                    writer.add_scalar('Training/Score_MA10', np.mean(scores[-10:]), ep)
                    writer.add_scalar('Training/Lines_MA10', np.mean(lines[-10:]), ep)
                if len(rewards) >= 50:
                    writer.add_scalar('Training/Reward_MA50', np.mean(rewards[-50:]), ep)
                    writer.add_scalar('Training/Score_MA50', np.mean(scores[-50:]), ep)

            # Actualizar mejor episodio
            if episode_score > best_episode['score']:
                best_episode = {
                    'episode': ep + 1,
                    'score': episode_score,
                    'lines': total_lines,
                    'actions': episode_actions.copy(),
                    'reward': total_r
                }
                print(f"  🌟 ¡Nuevo mejor episodio! Score: {episode_score}, Líneas: {total_lines}")

                # Guardar mejor modelo
                agent.save(f"{run_dir}/checkpoints/best_model.pth")

            # Logging en consola
            if (ep + 1) % 10 == 0 or visualize:
                elapsed = time.time() - start_time
                eps_per_sec = (ep + 1) / elapsed
                eta = (episodes - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0

                print(f"Ep {ep+1}/{episodes} | Reward: {total_r:.1f} | Score: {episode_score} | Lines: {total_lines} | "
                      f"Loss: {avg_loss:.4f} | ε: {agent.epsilon:.3f} | "
                      f"ETA: {int(eta//60)}m {int(eta%60)}s")

            # Guardar checkpoint periódico
            if (ep + 1) % checkpoint_interval == 0:
                checkpoint_path = f"{run_dir}/checkpoints/model_ep{ep+1}.pth"
                agent.save(checkpoint_path)
                print(f"  💾 Checkpoint guardado: {checkpoint_path}")
        
        print("\n" + "="*70)
        print("¡ENTRENAMIENTO COMPLETADO!")
        print("="*70)

        # Guardar modelo final
        if save_model:
            model_path = "models/dqn_tetris_final.pth"
            agent.save(model_path)
            agent.save(f"{run_dir}/final_model.pth")
            print(f"✓ Modelo final guardado en {model_path}")

        # Cerrar TensorBoard
        if writer:
            writer.close()

        # Generar gráficos detallados
        print("\nGenerando gráficos de análisis...")
        import matplotlib.pyplot as plt

        # Gráfico 1: Rewards y Scores
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        episodes_arr = np.arange(1, len(rewards) + 1)

        ax1.plot(episodes_arr, rewards, alpha=0.2, label='Reward', color='blue')
        if len(rewards) >= 10:
            ma10 = np.convolve(rewards, np.ones(10)/10, mode='valid')
            ax1.plot(np.arange(len(ma10))+1, ma10, label='MA-10', color='blue', linewidth=2)
        if len(rewards) >= 50:
            ma50 = np.convolve(rewards, np.ones(50)/50, mode='valid')
            ax1.plot(np.arange(len(ma50))+1, ma50, label='MA-50', color='darkblue', linewidth=2)
        ax1.set_ylabel('Reward')
        ax1.set_title('DQN Training Progress - Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(episodes_arr, scores, alpha=0.2, label='Score', color='green')
        if len(scores) >= 10:
            ma10 = np.convolve(scores, np.ones(10)/10, mode='valid')
            ax2.plot(np.arange(len(ma10))+1, ma10, label='MA-10', color='green', linewidth=2)
        if len(scores) >= 50:
            ma50 = np.convolve(scores, np.ones(50)/50, mode='valid')
            ax2.plot(np.arange(len(ma50))+1, ma50, label='MA-50', color='darkgreen', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.set_title('Game Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{run_dir}/rewards_scores.png", dpi=150)
        plt.close()

        # Gráfico 2: Líneas y Piezas
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(episodes_arr, lines, alpha=0.2, label='Lines', color='purple')
        if len(lines) >= 50:
            ma50 = np.convolve(lines, np.ones(50)/50, mode='valid')
            ax1.plot(np.arange(len(ma50))+1, ma50, label='MA-50', color='darkviolet', linewidth=2)
        ax1.set_ylabel('Lines Cleared')
        ax1.set_title('Lines Cleared per Episode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(episodes_arr, pieces_placed, alpha=0.2, label='Pieces', color='orange')
        if len(pieces_placed) >= 50:
            ma50 = np.convolve(pieces_placed, np.ones(50)/50, mode='valid')
            ax2.plot(np.arange(len(ma50))+1, ma50, label='MA-50', color='darkorange', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Pieces Placed')
        ax2.set_title('Pieces Placed per Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{run_dir}/lines_pieces.png", dpi=150)
        plt.close()

        # Gráfico 3: Loss y Epsilon
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(episodes_arr, losses, alpha=0.2, label='Loss', color='red')
        if len(losses) >= 50:
            ma50 = np.convolve(losses, np.ones(50)/50, mode='valid')
            ax1.plot(np.arange(len(ma50))+1, ma50, label='MA-50', color='darkred', linewidth=2)
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        epsilons = [config['epsilon_start'] * (config['epsilon_decay'] ** i) for i in range(len(rewards))]
        epsilons = [max(e, config['epsilon_min']) for e in epsilons]
        ax2.plot(episodes_arr, epsilons, label='Epsilon', color='teal', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Exploration Rate (Epsilon)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{run_dir}/loss_epsilon.png", dpi=150)
        plt.close()

        # Gráfico 4: Q-values
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(episodes_arr, q_values_avg, alpha=0.2, label='Avg Q-value', color='cyan')
        if len(q_values_avg) >= 50:
            ma50 = np.convolve(q_values_avg, np.ones(50)/50, mode='valid')
            ax.plot(np.arange(len(ma50))+1, ma50, label='MA-50', color='darkcyan', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Q-value')
        ax.set_title('Average Q-values Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{run_dir}/q_values.png", dpi=150)
        plt.close()

        # Generar resumen de estadísticas
        summary = {
            'training_time': time.time() - start_time,
            'total_episodes': episodes,
            'best_episode': best_episode['episode'],
            'best_score': best_episode['score'],
            'best_lines': best_episode['lines'],
            'avg_reward_last_50': float(np.mean(rewards[-50:])),
            'avg_score_last_50': float(np.mean(scores[-50:])),
            'avg_lines_last_50': float(np.mean(lines[-50:])),
            'avg_pieces_last_50': float(np.mean(pieces_placed[-50:])),
            'final_epsilon': agent.epsilon,
            'max_reward': float(np.max(rewards)),
            'max_score': float(np.max(scores)),
            'max_lines': int(np.max(lines))
        }

        with open(f"{run_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=4)

        # Imprimir resumen
        print("\n" + "="*70)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("="*70)
        print(f"Tiempo total: {int(summary['training_time']//60)}m {int(summary['training_time']%60)}s")
        print(f"\nMejor episodio: {best_episode['episode']}")
        print(f"  - Score: {best_episode['score']}")
        print(f"  - Líneas: {best_episode['lines']}")
        print(f"  - Reward: {best_episode['reward']:.1f}")
        print(f"\nPromedios (últimos 50 episodios):")
        print(f"  - Reward: {summary['avg_reward_last_50']:.1f}")
        print(f"  - Score: {summary['avg_score_last_50']:.1f}")
        print(f"  - Líneas: {summary['avg_lines_last_50']:.1f}")
        print(f"  - Piezas: {summary['avg_pieces_last_50']:.1f}")
        print(f"\nMáximos alcanzados:")
        print(f"  - Score: {summary['max_score']}")
        print(f"  - Líneas: {summary['max_lines']}")
        print(f"\nEpsilon final: {summary['final_epsilon']:.4f}")
        print(f"\n📁 Resultados guardados en: {run_dir}")
        print("="*70)
        
        # Reproducir mejor episodio
        if replay_best and best_episode['episode'] > 0:
            print("\n" + "="*70)
            print(f"  🏆 REPRODUCIENDO MEJOR EPISODIO")
            print("="*70)
            print(f"Episodio: {best_episode['episode']}/{episodes}")
            print(f"Score: {best_episode['score']}")
            print(f"Líneas: {best_episode['lines']}")
            print(f"Reward: {best_episode['reward']:.1f}")
            print("="*70)
            input("\nPresiona ENTER para ver la reproducción...")
            
            replay_episode(best_episode['actions'], delay_ms=80)
        
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
        if save_model:
            model_path = "models/dqn_tetris_interrupted.pth"
            agent.save(model_path)
            print(f"Modelo parcial guardado en {model_path}")
    
    finally:
        if visual_env is not None:
            visual_env.close()
        env.close()

if __name__ == "__main__":
    # Configuración de entrenamiento
    run(episodes=500, visual_interval=50, delay_ms=30, save_model=True, replay_best=True)