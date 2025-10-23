"""
Test del ambiente de Tetris
"""
import sys
sys.path.append('.')

from envs.tetris_env import TetrisEnv

print("Probando ambiente de Tetris...\n")

# Crear ambiente
env = TetrisEnv(rows=20, cols=10)
print(f"✓ Ambiente creado")
print(f"  Observación shape: {env.observation_space.shape}")
print(f"  Acciones disponibles: {env.action_space.n}")

# Test rápido
obs, info = env.reset()
total_reward = 0

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"\n✓ Juego terminado en step {step}")
        print(f"  Score: {info['score']}")
        print(f"  Líneas completadas: {info['lines']}")
        print(f"  Reward total: {total_reward}")
        break

env.close()
print("\n✓ Test completado exitosamente")