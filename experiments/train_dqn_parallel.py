import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import torch
from datetime import datetime
from envs.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from utils.parallel_env import ParallelEnv

def make_env():
    # Crea entorno (top-level para pickle en Windows)
    return TetrisEnv(use_action_masking=True)

def train_dqn_parallel(episodes=2000, num_parallel_envs=8, save_interval=500):
    # Entrena DQN con multiples entornos en paralelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/dqn_parallel_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print("="*70)
    print("ENTRENAMIENTO DQN PARALELO")
    print("="*70)

    # Verificar CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA no disponible, usando CPU")

    print(f"Entornos paralelos: {num_parallel_envs}")
    print(f"Episodios totales: {episodes}")
    print(f"Resultados: {results_dir}")
    print("="*70)

    # Crear entornos paralelos
    parallel_envs = ParallelEnv(make_env, num_envs=num_parallel_envs)

    # Crear agente con un entorno de referencia
    single_env = make_env()
    agent = DQNAgent(
        single_env,
        lr=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99995,  # Decay MUY lento para mas exploracion
        buffer_size=200000,  # Buffer mas grande
        batch_size=256,  # Batch mas grande para mejor aprendizaje
        target_update=5000,
        use_double_dqn=True
    )

    print(f"\nConfiguracion del agente:")
    print(f"  Device: {agent.device}")
    print(f"  Learning rate: 0.00025")
    print(f"  Epsilon decay: 0.99995 (muy lento)")
    print(f"  Buffer size: 200000")
    print(f"  Batch size: 256 (optimizado para GPU)")
    print(f"  Mixed precision: FP16 {'Activado' if agent.scaler else 'Desactivado'}")
    print(f"  Target update: 5000 pasos")
    print(f"  Double DQN: True")
    print(f"  Action masking: True")

    # Metricas
    metrics = {
        'scores': [],
        'lines': [],
        'pieces': [],
        'rewards': [],
        'losses': [],
        'computation_time': []
    }

    # Estado inicial de los entornos
    obs_batch, _ = parallel_envs.reset()
    episode_count = 0
    episode_rewards = [0.0] * num_parallel_envs
    episode_steps = [0] * num_parallel_envs
    episode_start_times = [time.time()] * num_parallel_envs

    print(f"\nIniciando entrenamiento...")
    print("-"*70)

    overall_start = time.time()

    while episode_count < episodes:
        # Obtener acciones validas para cada entorno
        valid_actions_batch = parallel_envs.get_valid_actions()

        # Seleccionar acciones usando batch inference
        actions = agent.select_actions_batch(
            obs_batch,
            training=True,
            valid_actions_list=valid_actions_batch
        )

        # Ejecutar step en paralelo
        next_obs_batch, rewards, terminateds, truncateds, infos = parallel_envs.step(actions)

        # Procesar cada entorno
        for i in range(num_parallel_envs):
            # Almacenar transicion
            agent.store_transition(
                obs_batch[i],
                actions[i],
                rewards[i],
                next_obs_batch[i],
                terminateds[i]
            )

            episode_rewards[i] += rewards[i]
            episode_steps[i] += 1

            # Si el episodio termino
            if terminateds[i] or truncateds[i]:
                episode_time = time.time() - episode_start_times[i]

                metrics['scores'].append(infos[i]['score'])
                metrics['lines'].append(infos[i]['lines'])
                metrics['pieces'].append(episode_steps[i])
                metrics['rewards'].append(episode_rewards[i])
                metrics['computation_time'].append(episode_time)

                episode_count += 1

                # Mostrar progreso cada 50 episodios
                if episode_count % 50 == 0:
                    avg_score = np.mean(metrics['scores'][-50:])
                    avg_lines = np.mean(metrics['lines'][-50:])
                    avg_reward = np.mean(metrics['rewards'][-50:])
                    avg_time = np.mean(metrics['computation_time'][-50:])
                    avg_loss = np.mean(metrics['losses'][-50:]) if metrics['losses'] else 0

                    # Calcular ETA
                    elapsed = time.time() - overall_start
                    eta = (elapsed / episode_count) * (episodes - episode_count)

                    print(f"DQN | Ep {episode_count}/{episodes} | "
                          f"{elapsed/60:.1f}min | ETA: {eta/60:.1f}min | "
                          f"Score: {avg_score:.1f} | Lines: {avg_lines:.1f} | "
                          f"ε={agent.epsilon:.3f}")

                # Guardar modelo intermedio
                if episode_count % save_interval == 0:
                    model_path = os.path.join(results_dir, f"dqn_ep{episode_count}.pth")
                    agent.save(model_path)
                    print(f"  Modelo guardado: {model_path}")

                # Resetear tracking de este entorno
                episode_rewards[i] = 0.0
                episode_steps[i] = 0
                episode_start_times[i] = time.time()

                if episode_count >= episodes:
                    break

        # Entrenar agente (4 pasos por cada step de entorno para mejor aprovechamiento)
        for _ in range(4):
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                if loss > 0:
                    metrics['losses'].append(loss)

        # Actualizar observaciones
        obs_batch = next_obs_batch

    # Cerrar entornos paralelos
    parallel_envs.close()

    total_time = time.time() - overall_start

    # Guardar modelo final
    model_path = os.path.join(results_dir, "dqn_parallel_final.pth")
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
        'pieces': metrics['pieces'],
        'reward': metrics['rewards']
    })
    csv_path = os.path.join(results_dir, 'dqn_parallel_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado: {csv_path}")

    # Guardar configuracion
    config = {
        'episodes': episodes,
        'num_parallel_envs': num_parallel_envs,
        'learning_rate': 0.00025,
        'buffer_size': 200000,
        'batch_size': 256,
        'target_update': 5000,
        'epsilon_decay': 0.99995,
        'epsilon_min': 0.1,
        'use_double_dqn': True,
        'use_action_masking': True,
        'total_training_time': total_time,
        'avg_time_per_episode': total_time / episodes
    }

    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Resumen final
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Tiempo total: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
    print(f"Tiempo por episodio: {total_time/episodes:.2f}s")
    print(f"Speedup vs secuencial: ~{num_parallel_envs * 0.7:.1f}x")
    print(f"\nResultados finales (ultimos 100 episodios):")
    print(f"  Score promedio: {np.mean(metrics['scores'][-100:]):.1f}")
    print(f"  Lineas promedio: {np.mean(metrics['lines'][-100:]):.1f}")
    print(f"  Piezas promedio: {np.mean(metrics['pieces'][-100:]):.1f}")
    print(f"  Score maximo: {np.max(metrics['scores']):.1f}")
    print(f"  Lineas maximo: {np.max(metrics['lines'])}")
    print(f"\nModelo guardado en: {model_path}")
    print(f"Metricas guardadas en: {metrics_path}")
    print(f"Configuracion guardada en: {config_path}")

    return agent, metrics

if __name__ == "__main__":
    # Entrenar con 8 entornos en paralelo
    agent, metrics = train_dqn_parallel(
        episodes=2000,
        num_parallel_envs=8,
        save_interval=500
    )

    print("\nEntrenamiento paralelo finalizado correctamente")
