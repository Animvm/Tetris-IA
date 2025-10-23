import sys
sys.path.append('.')

# train_heuristic.py
from envs.tetris_env import TetrisEnv
from agents.heuristic_agent import HeuristicAgent
from utils.plotting import plot_metrics

def run(episodes=200):
    env = TetrisEnv()
    agent = HeuristicAgent(env)
    rewards = []
    lines = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0
        total_lines = 0
        while not done:
            a = agent.best_action(obs)
            obs, r, done, truncated, info = env.step(a)
            total_r += r
            total_lines = info.get("lines", total_lines)
        rewards.append(total_r)
        lines.append(total_lines)
        if (ep+1) % 10 == 0:
            print(f"Ep {ep+1}/{episodes} reward {total_r} lines {total_lines}")
    plot_metrics(rewards, lines, savepath="results/heuristic_progress.png", title="Heuristic agent")

if __name__ == "__main__":
    run(episodes=200)
