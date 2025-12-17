import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import torch
from datetime import datetime
from envs.tetris_env import TetrisEnv
from agents.hybrid_dqn_agent import HybridDQNAgent
from agents.expert_generator import MCTSExpertGenerator

def train_hybrid(episodes=1500, pretrain_episodes=500, expert_data_episodes=100,
                 use_existing_data=False, expert_data_path=None):
    # Entrena agente hibrido: pre-training con imitacion + fine-tuning con RL
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/hybrid_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print("="*70)
    print("ENTRENAMIENTO HIBRIDO MCTS-DQN")
    print("="*70)

    # Verificar CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA no disponible, usando CPU")

    print(f"Resultados: {results_dir}")
    print("="*70)

    # Paso 1: Generar o cargar datos expertos
    if use_existing_data and expert_data_path and os.path.exists(expert_data_path):
        print(f"\nUsando datos expertos existentes: {expert_data_path}")
    else:
        print(f"\nPASO 1: Generando datos expertos con MCTS")
        print(f"Episodios a generar: {expert_data_episodes}")
        print(f"Simulaciones MCTS: 50 por accion (reducido para velocidad)")
        print("-"*70)

        env_expert = TetrisEnv(use_action_masking=True)
        generator = MCTSExpertGenerator(env_expert, num_simulations=50,
                                       save_dir="data/expert_demos")

        start_time = time.time()
        dataset, expert_data_path = generator.generate_dataset(
            num_episodes=expert_data_episodes,
            min_score=50
        )
        gen_time = time.time() - start_time

        print(f"\nGeneracion completada en {gen_time:.1f} segundos")
        print(f"Tiempo promedio por episodio: {gen_time/expert_data_episodes:.1f}s")

    # Paso 2: Crear agente hibrido
    print(f"\nPASO 2: Creando agente hibrido")
    print("-"*70)

    env = TetrisEnv(use_action_masking=True)
    agent = HybridDQNAgent(
        env,
        expert_data_path=expert_data_path,
        imitation_weight=1.0,  # Comenzar con imitacion fuerte
        lr=0.0001,
        buffer_size=50000,
        batch_size=64,
        target_update=1000,
        use_double_dqn=True
    )

    print(f"Configuracion del agente:")
    print(f"  Device: {agent.device}")
    print(f"  Imitation weight inicial: {agent.imitation_weight}")
    print(f"  Buffer experto: {len(agent.expert_memory)} transiciones")
    print(f"  Buffer propio: {len(agent.memory)} transiciones")
    print(f"  Learning rate: {agent.optimizer.param_groups[0]['lr']}")
    print(f"  Mixed precision: FP16 {'Activado' if agent.scaler else 'Desactivado'}")

    # Paso 3: Pre-entrenamiento (imitacion pura)
    print(f"\nPASO 3: Pre-entrenamiento por imitacion")
    print(f"Episodios: {pretrain_episodes}")
    print("-"*70)

    start_time = time.time()

    for ep in range(pretrain_episodes):
        # Multiples pasos de gradiente por episodio sin interactuar con ambiente
        for _ in range(10):
            if len(agent.expert_memory) >= agent.batch_size:
                total_loss, td_loss, im_loss = agent.train_step_hybrid()

        if (ep + 1) % 100 == 0:
            print(f"Pre-train {ep+1:4d}/{pretrain_episodes}: "
                  f"Total Loss={total_loss:.4f}, "
                  f"TD Loss={td_loss:.4f}, "
                  f"Imit Loss={im_loss:.4f}")

    pretrain_time = time.time() - start_time
    print(f"\nPre-entrenamiento completado en {pretrain_time:.1f} segundos")

    # Paso 4: Fine-tuning con RL
    print(f"\nPASO 4: Fine-tuning con RL")
    print(f"Episodios: {episodes}")
    print("-"*70)

    metrics = {
        'scores': [],
        'lines': [],
        'rewards': [],
        'total_losses': [],
        'td_losses': [],
        'imitation_losses': [],
        'imitation_weights': [],
        'epsilons': []
    }

    start_time = time.time()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_losses = {'total': [], 'td': [], 'imitation': []}

        while not done:
            # Obtener acciones validas
            valid_actions = env.get_valid_actions() if env.use_action_masking else None

            # Seleccionar accion
            action = agent.select_action(obs, training=True, valid_actions=valid_actions)

            # Ejecutar accion
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Almacenar en buffer propio
            agent.store_transition(obs, action, reward, next_obs, done)

            # Entrenar con loss hibrido
            total_loss, td_loss, im_loss = agent.train_step_hybrid()

            if total_loss > 0:
                episode_losses['total'].append(total_loss)
                episode_losses['td'].append(td_loss)
                episode_losses['imitation'].append(im_loss)

            total_reward += reward
            obs = next_obs

        # Registrar metricas
        metrics['scores'].append(info['score'])
        metrics['lines'].append(info['lines'])
        metrics['rewards'].append(total_reward)
        metrics['total_losses'].append(np.mean(episode_losses['total']) if episode_losses['total'] else 0)
        metrics['td_losses'].append(np.mean(episode_losses['td']) if episode_losses['td'] else 0)
        metrics['imitation_losses'].append(np.mean(episode_losses['imitation']) if episode_losses['imitation'] else 0)
        metrics['imitation_weights'].append(agent.imitation_weight)
        metrics['epsilons'].append(agent.epsilon)

        # Mostrar progreso
        if (ep + 1) % 50 == 0:
            avg_score = np.mean(metrics['scores'][-50:])
            avg_lines = np.mean(metrics['lines'][-50:])
            avg_loss = np.mean(metrics['total_losses'][-50:])

            print(f"Ep {ep+1:4d}/{episodes}: "
                  f"Score={info['score']:6.1f} (avg={avg_score:6.1f}), "
                  f"Lines={info['lines']:2d} (avg={avg_lines:4.1f}), "
                  f"Loss={avg_loss:.3f}, "
                  f"ε={agent.epsilon:.3f}, "
                  f"λ={agent.imitation_weight:.4f}")

    training_time = time.time() - start_time

    # Guardar modelo
    model_path = os.path.join(results_dir, "hybrid_model_final.pth")
    agent.save(model_path)

    # Guardar metricas
    import json
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f, indent=2)

    # Guardar metricas en CSV
    import pandas as pd
    df = pd.DataFrame({
        'episode': list(range(1, len(metrics['scores'])+1)),
        'score': metrics['scores'],
        'lines': metrics['lines'],
        'reward': metrics['rewards'],
        'total_loss': metrics['total_losses'],
        'imitation_weight': metrics['imitation_weights']
    })

    csv_path = os.path.join(results_dir, 'hybrid_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado: {csv_path}")

    # Resumen final
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Tiempo total: {training_time:.1f} segundos")
    print(f"Tiempo por episodio: {training_time/episodes:.2f}s")
    print(f"\nResultados finales (ultimos 100 episodios):")
    print(f"  Score promedio: {np.mean(metrics['scores'][-100:]):.1f}")
    print(f"  Lineas promedio: {np.mean(metrics['lines'][-100:]):.1f}")
    print(f"  Score maximo: {np.max(metrics['scores']):.1f}")
    print(f"  Lineas maximo: {np.max(metrics['lines'])}")
    print(f"\nModelo guardado en: {model_path}")
    print(f"Metricas guardadas en: {metrics_path}")

    return agent, metrics

if __name__ == "__main__":
    # Entrenar agente hibrido
    agent, metrics = train_hybrid(
        episodes=1500,
        pretrain_episodes=500,
        expert_data_episodes=30,  # Reducido de 100 a 30 para velocidad
        use_existing_data=False
    )

    print("\nEntrenamiento hibrido finalizado correctamente")
