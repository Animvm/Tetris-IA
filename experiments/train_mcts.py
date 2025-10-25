import sys
sys.path.append('.')

from envs.tetris_env import TetrisEnv
from agents.mcts_agent import MCTSAgent

def entrenar(episodios=10, simulaciones=100, visualizar_final=False):
    env = TetrisEnv(rows=20, cols=10)
    agente = MCTSAgent(env, num_simulaciones=simulaciones, max_profundidad=8)
    
    resultados = []
    
    print(f"Entrenando MCTS: {episodios} episodios\n")
    
    for ep in range(episodios):
        obs, _ = env.reset()
        terminado = False
        recompensa_total = 0
        
        while not terminado:
            accion = agente.seleccionar_accion(obs)
            obs, recompensa, term, trunc, info = env.step(accion)
            recompensa_total += recompensa
            terminado = term or trunc
        
        resultados.append({
            'puntaje': info['score'],
            'lineas': info['lines'],
            'recompensa': recompensa_total
        })
        
        print(f"Ep {ep+1}/{episodios} - Puntaje: {info['score']}, Lineas: {info['lines']}, Recompensa: {recompensa_total}")
    
    print("\nResumen:")
    puntaje_promedio = sum(r['puntaje'] for r in resultados) / len(resultados)
    lineas_promedio = sum(r['lineas'] for r in resultados) / len(resultados)
    print(f"Puntaje promedio: {puntaje_promedio:.1f}")
    print(f"Lineas promedio: {lineas_promedio:.1f}")
    
    env.close()
    
    if visualizar_final:
        print("\n=== Visualizando episodio final ===")
        visualizar_episodio(agente)

def visualizar_episodio(agente):
    import pygame
    
    env_visual = TetrisEnv(rows=20, cols=10, render_mode="human")
    agente_visual = MCTSAgent(env_visual, num_simulaciones=50, max_profundidad=5)
    
    obs, _ = env_visual.reset()
    env_visual.render()
    terminado = False
    pasos = 0
    
    print("Jugando episodio (cada movimiento toma varios segundos)...")
    
    try:
        while not terminado and pasos < 30:
            print(f"  Calculando movimiento {pasos+1}...")
            accion = agente_visual.seleccionar_accion(obs)
            obs, recompensa, term, trunc, info = env_visual.step(accion)
            terminado = term or trunc
            pasos += 1
            env_visual.render()
            pygame.time.wait(300)
        
        print(f"\nEpisodio visual terminado")
        print(f"Puntaje: {info['score']}, Lineas: {info['lines']}")
        print("Cierra la ventana para continuar...")
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env_visual.close()
                    return
            pygame.time.wait(100)
            
    except KeyboardInterrupt:
        print("\nInterrumpido")
    finally:
        env_visual.close()

if __name__ == "__main__":
    entrenar(episodios=10, simulaciones=100, visualizar_final=False)