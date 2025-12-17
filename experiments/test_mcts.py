import sys
sys.path.append('.')

from envs.tetris_env import TetrisEnv
from agents.mcts_agent import MCTSAgent

print("Probando agente MCTS\n")

env = TetrisEnv(rows=20, cols=10)
agente = MCTSAgent(env, num_simulaciones=50, max_profundidad=5)

obs, _ = env.reset()
terminado = False
recompensa_total = 0
pasos = 0

while not terminado and pasos < 20:
    accion = agente.seleccionar_accion(obs)
    obs, recompensa, term, trunc, info = env.step(accion)
    recompensa_total += recompensa
    terminado = term or trunc
    pasos += 1

    if pasos % 5 == 0:
        print(f"Paso {pasos}: Puntaje={info['score']}, Lineas={info['lines']}")

print(f"\nEpisodio terminado en {pasos} pasos")
print(f"Puntaje: {info['score']}")
print(f"Lineas: {info['lines']}")
print(f"Recompensa total: {recompensa_total}")

env.close()
