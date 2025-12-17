import sys
sys.path.append('.')

import time
from envs.tetris_env import TetrisEnv
from agents.heuristic_agent import HeuristicAgent

# ejecuta episodios con visualizacion
def run_visual(episodes=5, delay_ms=150):
    env = TetrisEnv(render_mode="human", cell_size=28)
    agent = HeuristicAgent(env)

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            env.render()
            done = False
            total = 0
            while not done:
                action = agent.best_action(obs)
                obs, r, done, truncated, info = env.step(action)
                total += r
                env.render()
                pygame_delay(delay_ms)
            print(f"Episode {ep+1}/{episodes} done. Score {info.get('score')} Lines {info.get('lines')}")
        # guardar resultados
        with open("results/visual_scores.txt", "a") as f:
            f.write(f"{ep+1},{info['score']},{info['lines']}\n")

    except KeyboardInterrupt:
        print("Interrumpido por usuario.")
    finally:
        env.close()

# pausa para visualizacion
def pygame_delay(ms):
    import pygame
    start = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start < ms:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        pygame.time.delay(10)

if __name__ == "__main__":
    run_visual(episodes=3, delay_ms=120)
