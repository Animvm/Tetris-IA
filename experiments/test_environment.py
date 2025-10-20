"""Test básico del ambiente de Tetris"""
import gym
import gym_tetris

# Crear ambiente
env = gym.make('TetrisA-v0')
print(f"Ambiente creado: TetrisA-v0")
print(f"Observación shape: {env.observation_space.shape}")
print(f"Acciones disponibles: {env.action_space.n}")

# Test rápido
obs = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        print(f"\nJuego terminado en step {step}")
        break

env.close()
print("Test completado exitosamente")