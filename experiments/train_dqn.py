import sys
sys.path.append('.')

# train_dqn.py
import time
import pygame
import numpy as np
from envs.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from utils.plotting import plot_metrics
import os

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

def run(episodes=500, visual_interval=50, delay_ms=50, save_model=True, replay_best=True):
    """
    Entrenar DQN con visualización periódica
    
    Args:
        episodes: número total de episodios
        visual_interval: cada cuántos episodios mostrar visualización
        delay_ms: delay entre pasos en visualización (ms)
        save_model: si guardar el modelo entrenado
        replay_best: si reproducir el mejor episodio al finalizar
    """
    # Crear directorios
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Entorno y agente
    env = TetrisEnv()
    agent = DQNAgent(env, lr=0.0001, gamma=0.99, epsilon=1.0, 
                     epsilon_decay=0.995, batch_size=32)
    
    # Entorno visual (se crea solo cuando se necesita)
    visual_env = None
    
    rewards = []
    lines = []
    losses = []
    
    # Tracking del mejor episodio
    best_episode = {
        'episode': 0,
        'score': -float('inf'),
        'lines': 0,
        'actions': []
    }
    
    print("Iniciando entrenamiento DQN...")
    print(f"Device: {agent.device}")
    
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
            episode_actions = []  # Guardar acciones del episodio
            
            if visualize:
                current_env.render()
            
            while not done:
                # Seleccionar acción
                action = agent.select_action(obs, training=True)
                episode_actions.append(action)  # Guardar acción
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
                
                if visualize:
                    current_env.render()
                    pygame_delay(delay_ms)
            
            rewards.append(total_r)
            lines.append(total_lines)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            losses.append(avg_loss)
            
            # Actualizar mejor episodio (basado en score del juego)
            episode_score = info.get('score', total_r)
            if episode_score > best_episode['score']:
                best_episode = {
                    'episode': ep + 1,
                    'score': episode_score,
                    'lines': total_lines,
                    'actions': episode_actions.copy(),
                    'reward': total_r
                }
                print(f"  🌟 ¡Nuevo mejor episodio! Score: {episode_score}, Líneas: {total_lines}")
            
            # Logging
            if (ep + 1) % 10 == 0 or visualize:
                print(f"Ep {ep+1}/{episodes} | Reward: {total_r:.1f} | Lines: {total_lines} | "
                      f"Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
            
            # Guardar progreso periódicamente
            if (ep + 1) % 100 == 0:
                with open("results/dqn_scores.txt", "a") as f:
                    f.write(f"{ep+1},{total_r},{total_lines},{avg_loss},{agent.epsilon}\n")
        
        print("\n¡Entrenamiento completado!")
        
        # Guardar modelo
        if save_model:
            model_path = "models/dqn_tetris.pth"
            agent.save(model_path)
            print(f"Modelo guardado en {model_path}")
        
        # Generar gráficos
        print("Generando gráficos...")
        plot_metrics(rewards, lines, window=50, 
                    savepath="results/dqn_progress.png", 
                    title="DQN Agent Training")
        
        # Gráfico de pérdida
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(losses, alpha=0.3, label='Loss per episode')
        if len(losses) >= 50:
            moving_avg = np.convolve(losses, np.ones(50)/50, mode='valid')
            plt.plot(np.arange(len(moving_avg))+1, moving_avg, label='Moving avg (50)')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('DQN Training Loss')
        plt.legend()
        plt.savefig("results/dqn_loss.png")
        plt.close()
        
        print(f"Resultados guardados en results/")
        print(f"Reward promedio (últimos 50): {np.mean(rewards[-50:]):.1f}")
        print(f"Líneas promedio (últimos 50): {np.mean(lines[-50:]):.1f}")
        
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