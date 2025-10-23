import sys
sys.path.append('.')

# visualize_run.py
import time
from envs.tetris_env import TetrisEnv
from agents.heuristic_agent import HeuristicAgent

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
                # small delay so user can see progress
                pygame_delay(delay_ms)
            print(f"Episode {ep+1}/{episodes} done. Score {info.get('score')} Lines {info.get('lines')}")
        # después del bucle principal
        with open("results/visual_scores.txt", "a") as f:
            f.write(f"{ep+1},{info['score']},{info['lines']}\n")

    except KeyboardInterrupt:
        print("Interrumpido por usuario.")
    finally:
        env.close()

def pygame_delay(ms):
    # small helper to keep the window responsive while waiting
    import pygame
    start = pygame.time.get_ticks()
    # process events while waiting
    while pygame.time.get_ticks() - start < ms:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        pygame.time.delay(10)

if __name__ == "__main__":
    run_visual(episodes=3, delay_ms=120)
    